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

#include "RefinementRules.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace svmp {

// ====================
// LineRefinementRule Implementation
// ====================

bool LineRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Line && level < 10;  // Max 10 levels
}

size_t LineRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Line) return 0;
  return 2;  // Line always splits into 2
}

RefinedElement LineRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {

  if (family != CellFamily::Line || vertices.size() != 2) {
    throw std::invalid_argument("Invalid element for line refinement");
  }

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = spec.pattern;

  // Create midpoint
  auto midpoint = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
  refined.new_vertices.push_back(midpoint);
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});

  // Create two child lines
  // Child 0: vertex 0 to midpoint
  refined.child_connectivity.push_back({0, 2});  // 2 is the new midpoint index

  // Child 1: midpoint to vertex 1
  refined.child_connectivity.push_back({2, 1});
  refined.child_families = {CellFamily::Line, CellFamily::Line};

  return refined;
}

std::vector<RefinementPattern> LineRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Line) return {};
  return {RefinementPattern::ISOTROPIC};
}

RefinementPattern LineRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::ISOTROPIC;
}

// ====================
// TriangleRefinementRule Implementation
// ====================

bool TriangleRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Triangle && level < 10;
}

size_t TriangleRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Triangle) return 0;

  switch (pattern) {
    case RefinementPattern::RED:
      return 4;
    case RefinementPattern::BLUE:
      return 3;
    case RefinementPattern::GREEN:
    case RefinementPattern::BISECTION:
      return 2;
    default:
      return 4;
  }
}

RefinedElement TriangleRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {

  if (family != CellFamily::Triangle || vertices.size() != 3) {
    throw std::invalid_argument("Invalid element for triangle refinement");
  }

  switch (spec.pattern) {
    case RefinementPattern::RED:
      return red_refine(vertices, level);
    case RefinementPattern::GREEN:
      return green_refine(vertices, spec.selector, level);
    case RefinementPattern::BLUE:
      return blue_refine(vertices, spec.selector, level);
    case RefinementPattern::BISECTION:
      return bisect(vertices, level);
    default:
      return red_refine(vertices, level);
  }
}

std::vector<RefinementPattern> TriangleRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Triangle) return {};
  return {RefinementPattern::RED, RefinementPattern::GREEN, RefinementPattern::BLUE, RefinementPattern::BISECTION};
}

RefinementPattern TriangleRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::RED;
}

RefinedElement TriangleRefinementRule::red_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::RED;

  // Create edge midpoints
  // Edge 0-1
  auto m01 = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
  // Edge 1-2
  auto m12 = RefinementUtils::edge_midpoint(vertices[1], vertices[2]);
  // Edge 2-0
  auto m20 = RefinementUtils::edge_midpoint(vertices[2], vertices[0]);

  // Store new vertices (indices 3, 4, 5)
  refined.new_vertices.push_back(m01);
  refined.new_vertices.push_back(m12);
  refined.new_vertices.push_back(m20);
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
  refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});
  refined.new_vertex_weights.push_back({{2, 0.5}, {0, 0.5}});

  // Create 4 child triangles
  // Original vertices: 0, 1, 2
  // New vertices: 3 (m01), 4 (m12), 5 (m20)

  // Corner triangles
  refined.child_connectivity.push_back({0, 3, 5});  // Triangle at vertex 0
  refined.child_connectivity.push_back({3, 1, 4});  // Triangle at vertex 1
  refined.child_connectivity.push_back({5, 4, 2});  // Triangle at vertex 2

  // Center triangle
  refined.child_connectivity.push_back({3, 4, 5});
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Triangle);

  return refined;
}

RefinedElement TriangleRefinementRule::green_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t edge_to_split,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::GREEN;

  // Split only one edge
  std::array<real_t, 3> midpoint;
  size_t v0 = 0;
  size_t v1 = 0;
  size_t v2 = 0;

  // Determine which edge to split and vertex ordering
  switch (edge_to_split) {
    case 0:  // Split edge 0-1
      v0 = 0; v1 = 1; v2 = 2;
      midpoint = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
      break;
    case 1:  // Split edge 1-2
      v0 = 1; v1 = 2; v2 = 0;
      midpoint = RefinementUtils::edge_midpoint(vertices[1], vertices[2]);
      break;
    case 2:  // Split edge 2-0
      v0 = 2; v1 = 0; v2 = 1;
      midpoint = RefinementUtils::edge_midpoint(vertices[2], vertices[0]);
      break;
    default:
      v0 = 0; v1 = 1; v2 = 2;
      midpoint = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
  }

  refined.new_vertices.push_back(midpoint);
  refined.new_vertex_weights.push_back({{v0, 0.5}, {v1, 0.5}});

  // Create 2 child triangles
  // New vertex index is 3
  refined.child_connectivity.push_back({v0, 3, v2});  // First child
  refined.child_connectivity.push_back({3, v1, v2});  // Second child
  refined.child_families = {CellFamily::Triangle, CellFamily::Triangle};

  return refined;
}

RefinedElement TriangleRefinementRule::blue_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t shared_vertex,
    size_t level) const {
  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::BLUE;

  const size_t s = shared_vertex % 3u;
  const size_t a = (s + 1u) % 3u;
  const size_t b = (s + 2u) % 3u;

  // New vertices: midpoint(s,a) then midpoint(s,b). Local indices 3 and 4.
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[s], vertices[a]));
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[s], vertices[b]));
  refined.new_vertex_weights.push_back({{s, 0.5}, {a, 0.5}});
  refined.new_vertex_weights.push_back({{s, 0.5}, {b, 0.5}});

  // Children:
  //  - small triangle at shared vertex
  //  - and two triangles completing the domain.
  refined.child_connectivity.push_back({s, 3, 4});
  refined.child_connectivity.push_back({3, a, b});
  refined.child_connectivity.push_back({4, 3, b});
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Triangle);

  return refined;
}

RefinedElement TriangleRefinementRule::bisect(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {

  // Find longest edge
  size_t longest_edge = RefinementUtils::find_longest_edge(vertices, CellFamily::Triangle);

  // Use green refinement on longest edge
  return green_refine(vertices, longest_edge, level);
}

// ====================
// QuadRefinementRule Implementation
// ====================

bool QuadRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Quad && level < 10;
}

size_t QuadRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Quad) return 0;

  switch (pattern) {
    case RefinementPattern::ISOTROPIC:
    case RefinementPattern::RED:
      return 4;
    case RefinementPattern::ANISOTROPIC:
      return 2;
    default:
      return 4;
  }
}

size_t QuadRefinementRule::num_children(CellFamily family, const RefinementSpec& spec) const {
  (void)spec.selector;
  return num_children(family, spec.pattern);
}

RefinedElement QuadRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {

  if (family != CellFamily::Quad || vertices.size() != 4) {
    throw std::invalid_argument("Invalid element for quad refinement");
  }

  switch (spec.pattern) {
    case RefinementPattern::ANISOTROPIC:
      return anisotropic_refine(vertices, spec.selector % 2u, level);
    default:
      return regular_refine(vertices, level);
  }
}

std::vector<RefinementPattern> QuadRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Quad) return {};
  return {RefinementPattern::ISOTROPIC, RefinementPattern::ANISOTROPIC};
}

RefinementPattern QuadRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::ISOTROPIC;
}

RefinedElement QuadRefinementRule::regular_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::ISOTROPIC;

  // Create edge midpoints
  auto m01 = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
  auto m12 = RefinementUtils::edge_midpoint(vertices[1], vertices[2]);
  auto m23 = RefinementUtils::edge_midpoint(vertices[2], vertices[3]);
  auto m30 = RefinementUtils::edge_midpoint(vertices[3], vertices[0]);

  // Create center point
  auto center = RefinementUtils::cell_center(vertices);

  // Store new vertices (indices 4, 5, 6, 7, 8)
  refined.new_vertices.push_back(m01);  // 4
  refined.new_vertices.push_back(m12);  // 5
  refined.new_vertices.push_back(m23);  // 6
  refined.new_vertices.push_back(m30);  // 7
  refined.new_vertices.push_back(center);  // 8
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
  refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});
  refined.new_vertex_weights.push_back({{2, 0.5}, {3, 0.5}});
  refined.new_vertex_weights.push_back({{3, 0.5}, {0, 0.5}});
  refined.new_vertex_weights.push_back({{0, 0.25}, {1, 0.25}, {2, 0.25}, {3, 0.25}});

  // Create 4 child quads
  // Original vertices: 0, 1, 2, 3
  // New vertices: 4 (m01), 5 (m12), 6 (m23), 7 (m30), 8 (center)

  refined.child_connectivity.push_back({0, 4, 8, 7});  // Bottom-left
  refined.child_connectivity.push_back({4, 1, 5, 8});  // Bottom-right
  refined.child_connectivity.push_back({8, 5, 2, 6});  // Top-right
  refined.child_connectivity.push_back({7, 8, 6, 3});  // Top-left
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Quad);

  return refined;
}

RefinedElement QuadRefinementRule::anisotropic_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t direction,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::ANISOTROPIC;

  if (direction == 0) {
    // Split in horizontal direction
    auto m01 = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
    auto m32 = RefinementUtils::edge_midpoint(vertices[3], vertices[2]);

    refined.new_vertices.push_back(m01);  // 4
    refined.new_vertices.push_back(m32);  // 5
    refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
    refined.new_vertex_weights.push_back({{3, 0.5}, {2, 0.5}});

    // Two child quads
    refined.child_connectivity.push_back({0, 4, 5, 3});  // Left
    refined.child_connectivity.push_back({4, 1, 2, 5});  // Right
  } else {
    // Split in vertical direction
    auto m03 = RefinementUtils::edge_midpoint(vertices[0], vertices[3]);
    auto m12 = RefinementUtils::edge_midpoint(vertices[1], vertices[2]);

    refined.new_vertices.push_back(m03);  // 4
    refined.new_vertices.push_back(m12);  // 5
    refined.new_vertex_weights.push_back({{0, 0.5}, {3, 0.5}});
    refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});

    // Two child quads
    refined.child_connectivity.push_back({0, 1, 5, 4});  // Bottom
    refined.child_connectivity.push_back({4, 5, 2, 3});  // Top
  }
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Quad);

  return refined;
}

// ====================
// TetrahedronRefinementRule Implementation
// ====================

bool TetrahedronRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Tetra && level < 10;
}

size_t TetrahedronRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Tetra) return 0;

  switch (pattern) {
    case RefinementPattern::RED:
      return 8;
    case RefinementPattern::GREEN:
      return 4;
    case RefinementPattern::BISECTION:
      return 2;
    default:
      return 8;
  }
}

size_t TetrahedronRefinementRule::num_children(CellFamily family, const RefinementSpec& spec) const {
  (void)spec.selector;
  return num_children(family, spec.pattern);
}

RefinedElement TetrahedronRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {

  if (family != CellFamily::Tetra || vertices.size() != 4) {
    throw std::invalid_argument("Invalid element for tetrahedron refinement");
  }

  switch (spec.pattern) {
    case RefinementPattern::GREEN:
      return face_green_refine(vertices, spec.selector % 4u, level);
    case RefinementPattern::BISECTION:
      return bisect(vertices, RefinementUtils::find_longest_edge(vertices, family), level);
    default:
      return red_refine(vertices, level);
  }
}

std::vector<RefinementPattern> TetrahedronRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Tetra) return {};
  return {RefinementPattern::RED, RefinementPattern::GREEN, RefinementPattern::BISECTION};
}

RefinementPattern TetrahedronRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::RED;
}

RefinedElement TetrahedronRefinementRule::red_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::RED;

  // Create edge midpoints (6 edges)
  auto m01 = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
  auto m02 = RefinementUtils::edge_midpoint(vertices[0], vertices[2]);
  auto m03 = RefinementUtils::edge_midpoint(vertices[0], vertices[3]);
  auto m12 = RefinementUtils::edge_midpoint(vertices[1], vertices[2]);
  auto m13 = RefinementUtils::edge_midpoint(vertices[1], vertices[3]);
  auto m23 = RefinementUtils::edge_midpoint(vertices[2], vertices[3]);

  // Store new vertices (indices 4-9)
  refined.new_vertices.push_back(m01);  // 4
  refined.new_vertices.push_back(m02);  // 5
  refined.new_vertices.push_back(m03);  // 6
  refined.new_vertices.push_back(m12);  // 7
  refined.new_vertices.push_back(m13);  // 8
  refined.new_vertices.push_back(m23);  // 9
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
  refined.new_vertex_weights.push_back({{0, 0.5}, {2, 0.5}});
  refined.new_vertex_weights.push_back({{0, 0.5}, {3, 0.5}});
  refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});
  refined.new_vertex_weights.push_back({{1, 0.5}, {3, 0.5}});
  refined.new_vertex_weights.push_back({{2, 0.5}, {3, 0.5}});

  // Create 8 child tetrahedra
  // Corner tets (at original vertices)
  refined.child_connectivity.push_back({0, 4, 5, 6});  // At vertex 0
  refined.child_connectivity.push_back({4, 1, 7, 8});  // At vertex 1
  refined.child_connectivity.push_back({5, 7, 2, 9});  // At vertex 2
  refined.child_connectivity.push_back({6, 8, 9, 3});  // At vertex 3

  // Octahedron subdivision (4 tets)
  // The inner octahedron formed by the 6 edge midpoints
  // needs to be split into 4 tetrahedra
  refined.child_connectivity.push_back({4, 5, 6, 8});
  refined.child_connectivity.push_back({4, 5, 8, 7});
  refined.child_connectivity.push_back({5, 6, 8, 9});
  refined.child_connectivity.push_back({5, 7, 9, 8});
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Tetra);

  return refined;
}

RefinedElement TetrahedronRefinementRule::face_green_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t opposite_vertex,
    size_t level) const {
  if (opposite_vertex > 3) {
    throw std::invalid_argument("face_green_refine: opposite_vertex out of range");
  }

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::GREEN;

  const size_t ov = opposite_vertex;
  std::array<size_t, 3> face{};
  size_t k = 0;
  for (size_t i = 0; i < 4; ++i) {
    if (i == ov) continue;
    face[k++] = i;
  }

  const size_t a = face[0];
  const size_t b = face[1];
  const size_t c = face[2];

  // Face-edge midpoints: (a,b), (b,c), (c,a). New local indices 4,5,6.
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[a], vertices[b]));
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[b], vertices[c]));
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[c], vertices[a]));
  refined.new_vertex_weights.push_back({{a, 0.5}, {b, 0.5}});
  refined.new_vertex_weights.push_back({{b, 0.5}, {c, 0.5}});
  refined.new_vertex_weights.push_back({{c, 0.5}, {a, 0.5}});

  // Face refined into 4 triangles, each connected to ov.
  const size_t mab = 4;
  const size_t mbc = 5;
  const size_t mca = 6;

  refined.child_connectivity.push_back({ov, a, mab, mca});
  refined.child_connectivity.push_back({ov, mab, b, mbc});
  refined.child_connectivity.push_back({ov, mca, mbc, c});
  refined.child_connectivity.push_back({ov, mab, mbc, mca});

  auto get_point = [&](size_t idx) -> std::array<real_t, 3> {
    if (idx < vertices.size()) {
      return vertices[idx];
    }
    const size_t new_idx = idx - vertices.size();
    if (new_idx >= refined.new_vertices.size()) {
      return std::array<real_t, 3>{0.0, 0.0, 0.0};
    }
    return refined.new_vertices[new_idx];
  };

  auto signed_det = [&](const std::vector<size_t>& conn) -> real_t {
    const auto p0 = get_point(conn[0]);
    const auto p1 = get_point(conn[1]);
    const auto p2 = get_point(conn[2]);
    const auto p3 = get_point(conn[3]);

    const std::array<real_t, 3> e1 = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const std::array<real_t, 3> e2 = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    const std::array<real_t, 3> e3 = {p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};

    return e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0]) +
           e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
  };

  const real_t det_parent = signed_det({0, 1, 2, 3});
  if (std::abs(det_parent) > static_cast<real_t>(0)) {
    for (auto& child : refined.child_connectivity) {
      const real_t det_child = signed_det(child);
      if (det_child * det_parent < static_cast<real_t>(0)) {
        std::swap(child[2], child[3]);
      }
    }
  }
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Tetra);

  return refined;
}

RefinedElement TetrahedronRefinementRule::bisect(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t longest_edge,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::BISECTION;

  // Bisection splits the tetrahedron along its longest edge
  // This creates 2 child tetrahedra

  // Determine vertices of longest edge
  size_t v0 = 0;
  size_t v1 = 0;
  size_t v2 = 0;
  size_t v3 = 0;
  switch (longest_edge) {
    case 0: v0 = 0; v1 = 1; v2 = 2; v3 = 3; break;
    case 1: v0 = 0; v1 = 2; v2 = 1; v3 = 3; break;
    case 2: v0 = 0; v1 = 3; v2 = 1; v3 = 2; break;
    case 3: v0 = 1; v1 = 2; v2 = 0; v3 = 3; break;
    case 4: v0 = 1; v1 = 3; v2 = 0; v3 = 2; break;
    case 5: v0 = 2; v1 = 3; v2 = 0; v3 = 1; break;
    default: v0 = 0; v1 = 1; v2 = 2; v3 = 3;
  }

  auto midpoint = RefinementUtils::edge_midpoint(vertices[v0], vertices[v1]);
  refined.new_vertices.push_back(midpoint);  // Index 4
  refined.new_vertex_weights.push_back({{v0, 0.5}, {v1, 0.5}});

  // Create 2 child tets
  refined.child_connectivity.push_back({v0, 4, v2, v3});  // First child
  refined.child_connectivity.push_back({4, v1, v2, v3});  // Second child
  refined.child_families = {CellFamily::Tetra, CellFamily::Tetra};

  return refined;
}

// ====================
// HexahedronRefinementRule Implementation
// ====================

bool HexahedronRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Hex && level < 10;
}

size_t HexahedronRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Hex) return 0;

  switch (pattern) {
    case RefinementPattern::ISOTROPIC:
    case RefinementPattern::RED:
      return 8;
    case RefinementPattern::ANISOTROPIC:
      return 4;
    default:
      return 8;
  }
}

size_t HexahedronRefinementRule::num_children(CellFamily family, const RefinementSpec& spec) const {
  if (family != CellFamily::Hex) return 0;
  if (spec.pattern != RefinementPattern::ANISOTROPIC) {
    return num_children(family, spec.pattern);
  }
  const std::uint32_t axis_mask = spec.selector;
  const unsigned bits = ((axis_mask & 1u) ? 1u : 0u) + ((axis_mask & 2u) ? 1u : 0u) + ((axis_mask & 4u) ? 1u : 0u);
  if (bits == 0u) {
    return 8;
  }
  return static_cast<size_t>(1u) << bits;
}

RefinedElement HexahedronRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {

  if (family != CellFamily::Hex || vertices.size() != 8) {
    throw std::invalid_argument("Invalid element for hexahedron refinement");
  }

  switch (spec.pattern) {
    case RefinementPattern::ANISOTROPIC:
      return anisotropic_refine(vertices, spec.selector, level);
    default:
      return regular_refine(vertices, level);
  }
}

std::vector<RefinementPattern> HexahedronRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Hex) return {};
  return {RefinementPattern::ISOTROPIC, RefinementPattern::ANISOTROPIC};
}

RefinementPattern HexahedronRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::ISOTROPIC;
}

RefinedElement HexahedronRefinementRule::regular_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::ISOTROPIC;

  // Hexahedron vertex numbering:
  // Bottom: 0, 1, 2, 3
  // Top: 4, 5, 6, 7

  // Create edge midpoints (12 edges)
  std::vector<std::array<real_t, 3>> edge_mids;
  // Bottom edges
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[0], vertices[1]));  // 8
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[1], vertices[2]));  // 9
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[2], vertices[3]));  // 10
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[3], vertices[0]));  // 11
  // Top edges
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[4], vertices[5]));  // 12
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[5], vertices[6]));  // 13
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[6], vertices[7]));  // 14
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[7], vertices[4]));  // 15
  // Vertical edges
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[0], vertices[4]));  // 16
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[1], vertices[5]));  // 17
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[2], vertices[6]));  // 18
  edge_mids.push_back(RefinementUtils::edge_midpoint(vertices[3], vertices[7]));  // 19

  // Create face centers (6 faces)
  std::vector<std::array<real_t, 3>> face_centers;
  // Bottom and top
  face_centers.push_back(RefinementUtils::face_center({vertices[0], vertices[1], vertices[2], vertices[3]}));  // 20
  face_centers.push_back(RefinementUtils::face_center({vertices[4], vertices[5], vertices[6], vertices[7]}));  // 21
  // Front and back
  face_centers.push_back(RefinementUtils::face_center({vertices[0], vertices[1], vertices[5], vertices[4]}));  // 22
  face_centers.push_back(RefinementUtils::face_center({vertices[3], vertices[2], vertices[6], vertices[7]}));  // 23
  // Left and right
  face_centers.push_back(RefinementUtils::face_center({vertices[0], vertices[3], vertices[7], vertices[4]}));  // 24
  face_centers.push_back(RefinementUtils::face_center({vertices[1], vertices[2], vertices[6], vertices[5]}));  // 25

  // Create cell center
  auto center = RefinementUtils::cell_center(vertices);  // 26

  // Store all new vertices
  for (size_t i = 0; i < edge_mids.size(); ++i) refined.new_vertices.push_back(edge_mids[i]);
  for (size_t i = 0; i < face_centers.size(); ++i) refined.new_vertices.push_back(face_centers[i]);
  refined.new_vertices.push_back(center);

  // Edge-midpoint weights (12)
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
  refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});
  refined.new_vertex_weights.push_back({{2, 0.5}, {3, 0.5}});
  refined.new_vertex_weights.push_back({{3, 0.5}, {0, 0.5}});
  refined.new_vertex_weights.push_back({{4, 0.5}, {5, 0.5}});
  refined.new_vertex_weights.push_back({{5, 0.5}, {6, 0.5}});
  refined.new_vertex_weights.push_back({{6, 0.5}, {7, 0.5}});
  refined.new_vertex_weights.push_back({{7, 0.5}, {4, 0.5}});
  refined.new_vertex_weights.push_back({{0, 0.5}, {4, 0.5}});
  refined.new_vertex_weights.push_back({{1, 0.5}, {5, 0.5}});
  refined.new_vertex_weights.push_back({{2, 0.5}, {6, 0.5}});
  refined.new_vertex_weights.push_back({{3, 0.5}, {7, 0.5}});
  // Face-center weights (6)
  refined.new_vertex_weights.push_back({{0, 0.25}, {1, 0.25}, {2, 0.25}, {3, 0.25}});
  refined.new_vertex_weights.push_back({{4, 0.25}, {5, 0.25}, {6, 0.25}, {7, 0.25}});
  refined.new_vertex_weights.push_back({{0, 0.25}, {1, 0.25}, {5, 0.25}, {4, 0.25}});
  refined.new_vertex_weights.push_back({{3, 0.25}, {2, 0.25}, {6, 0.25}, {7, 0.25}});
  refined.new_vertex_weights.push_back({{0, 0.25}, {3, 0.25}, {7, 0.25}, {4, 0.25}});
  refined.new_vertex_weights.push_back({{1, 0.25}, {2, 0.25}, {6, 0.25}, {5, 0.25}});
  // Cell-center weights
  refined.new_vertex_weights.push_back({{0, 0.125}, {1, 0.125}, {2, 0.125}, {3, 0.125},
                                       {4, 0.125}, {5, 0.125}, {6, 0.125}, {7, 0.125}});

  // Create 8 child hexahedra
  // Using the indexing: original (0-7), edge mids (8-19), face centers (20-25), center (26)

  // Bottom layer
  refined.child_connectivity.push_back({0, 8, 20, 11, 16, 22, 26, 24});   // Bottom-front-left
  refined.child_connectivity.push_back({8, 1, 9, 20, 22, 17, 25, 26});    // Bottom-front-right
  refined.child_connectivity.push_back({20, 9, 2, 10, 26, 25, 18, 23});   // Bottom-back-right
  refined.child_connectivity.push_back({11, 20, 10, 3, 24, 26, 23, 19});  // Bottom-back-left

  // Top layer
  refined.child_connectivity.push_back({16, 22, 26, 24, 4, 12, 21, 15});  // Top-front-left
  refined.child_connectivity.push_back({22, 17, 25, 26, 12, 5, 13, 21});  // Top-front-right
  refined.child_connectivity.push_back({26, 25, 18, 23, 21, 13, 6, 14});  // Top-back-right
  refined.child_connectivity.push_back({24, 26, 23, 19, 15, 21, 14, 7});  // Top-back-left
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Hex);

  return refined;
}

RefinedElement HexahedronRefinementRule::anisotropic_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    std::uint32_t axis_mask,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::ANISOTROPIC;

  // Axis bitmask X=1, Y=2, Z=4.
  const bool split_x = (axis_mask & 1u) != 0u;
  const bool split_y = (axis_mask & 2u) != 0u;
  const bool split_z = (axis_mask & 4u) != 0u;

  const std::vector<double> xs = split_x ? std::vector<double>{0.0, 0.5, 1.0} : std::vector<double>{0.0, 1.0};
  const std::vector<double> ys = split_y ? std::vector<double>{0.0, 0.5, 1.0} : std::vector<double>{0.0, 1.0};
  const std::vector<double> zs = split_z ? std::vector<double>{0.0, 0.5, 1.0} : std::vector<double>{0.0, 1.0};

  const auto corner_weights = [](double x, double y, double z) {
    const double xm = 1.0 - x;
    const double ym = 1.0 - y;
    const double zm = 1.0 - z;
    // Corner ordering matches the standard Hex vertex numbering in this file.
    return std::array<double, 8>{
        xm * ym * zm,  // 0 (0,0,0)
        x * ym * zm,   // 1 (1,0,0)
        x * y * zm,    // 2 (1,1,0)
        xm * y * zm,   // 3 (0,1,0)
        xm * ym * z,   // 4 (0,0,1)
        x * ym * z,    // 5 (1,0,1)
        x * y * z,     // 6 (1,1,1)
        xm * y * z     // 7 (0,1,1)
    };
  };

  // Map grid point (ix,iy,iz) to local vertex index in the refined element.
  // We treat the 8 original corners as parent-local indices 0..7; all other grid points are new.
  const int nx = static_cast<int>(xs.size());
  const int ny = static_cast<int>(ys.size());
  const int nz = static_cast<int>(zs.size());

  const auto is_corner = [&](int ix, int iy, int iz) {
    const bool cx = (ix == 0 || ix == nx - 1);
    const bool cy = (iy == 0 || iy == ny - 1);
    const bool cz = (iz == 0 || iz == nz - 1);
    return cx && cy && cz;
  };

  const auto corner_index = [&](int ix, int iy, int iz) -> size_t {
    const int xbit = (ix == nx - 1) ? 1 : 0;
    const int ybit = (iy == ny - 1) ? 1 : 0;
    const int zbit = (iz == nz - 1) ? 1 : 0;
    // Map (xbit,ybit,zbit) to standard corner id.
    // zbit=0: 0:(0,0),1:(1,0),2:(1,1),3:(0,1)
    // zbit=1: 4:(0,0),5:(1,0),6:(1,1),7:(0,1)
    if (zbit == 0) {
      if (ybit == 0) return xbit == 0 ? 0 : 1;
      return xbit == 0 ? 3 : 2;
    }
    if (ybit == 0) return xbit == 0 ? 4 : 5;
    return xbit == 0 ? 7 : 6;
  };

  const size_t kInvalid = std::numeric_limits<size_t>::max();
  std::vector<size_t> grid_to_local(static_cast<size_t>(nx * ny * nz), kInvalid);
  auto grid_flat = [&](int ix, int iy, int iz) { return (iz * ny + iy) * nx + ix; };

  // Assign local indices.
  size_t next_new_local = 8;  // new vertices start after parent corners
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int flat = grid_flat(ix, iy, iz);
        if (is_corner(ix, iy, iz)) {
          grid_to_local[static_cast<size_t>(flat)] = corner_index(ix, iy, iz);
        } else {
          grid_to_local[static_cast<size_t>(flat)] = next_new_local++;
        }
      }
    }
  }

  // Build new vertices list in ascending local index order.
  const size_t n_new = static_cast<size_t>(next_new_local - 8);
  refined.new_vertices.resize(n_new);
  refined.new_vertex_weights.resize(n_new);

  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        if (is_corner(ix, iy, iz)) continue;
        const size_t local = grid_to_local[static_cast<size_t>(grid_flat(ix, iy, iz))];
        const size_t out_idx = local - 8;
        const double x = xs[static_cast<size_t>(ix)];
        const double y = ys[static_cast<size_t>(iy)];
        const double z = zs[static_cast<size_t>(iz)];

        const auto w = corner_weights(x, y, z);
        std::array<real_t, 3> p{static_cast<real_t>(0.0), static_cast<real_t>(0.0), static_cast<real_t>(0.0)};
        std::vector<std::pair<size_t, double>> weights;
        for (size_t c = 0; c < 8; ++c) {
          if (w[c] == 0.0) continue;
          weights.emplace_back(c, w[c]);
          for (int d = 0; d < 3; ++d) {
            p[static_cast<size_t>(d)] += static_cast<real_t>(w[c]) * vertices[c][static_cast<size_t>(d)];
          }
        }
        refined.new_vertices[out_idx] = p;
        refined.new_vertex_weights[out_idx] = std::move(weights);
      }
    }
  }

  // Children: tensor-product subcells.
  const int cx = nx - 1;
  const int cy = ny - 1;
  const int cz = nz - 1;
  for (int kz = 0; kz < cz; ++kz) {
    for (int ky = 0; ky < cy; ++ky) {
      for (int kx = 0; kx < cx; ++kx) {
        const size_t v000 = grid_to_local[static_cast<size_t>(grid_flat(kx, ky, kz))];
        const size_t v100 = grid_to_local[static_cast<size_t>(grid_flat(kx + 1, ky, kz))];
        const size_t v110 = grid_to_local[static_cast<size_t>(grid_flat(kx + 1, ky + 1, kz))];
        const size_t v010 = grid_to_local[static_cast<size_t>(grid_flat(kx, ky + 1, kz))];
        const size_t v001 = grid_to_local[static_cast<size_t>(grid_flat(kx, ky, kz + 1))];
        const size_t v101 = grid_to_local[static_cast<size_t>(grid_flat(kx + 1, ky, kz + 1))];
        const size_t v111 = grid_to_local[static_cast<size_t>(grid_flat(kx + 1, ky + 1, kz + 1))];
        const size_t v011 = grid_to_local[static_cast<size_t>(grid_flat(kx, ky + 1, kz + 1))];
        refined.child_connectivity.push_back({v000, v100, v110, v010, v001, v101, v111, v011});
      }
    }
  }

  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Hex);
  return refined;
}

// ====================
// WedgeRefinementRule Implementation
// ====================

bool WedgeRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Wedge && level < 10;
}

size_t WedgeRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Wedge) return 0;
  switch (pattern) {
    case RefinementPattern::GREEN:
      return 4;
    case RefinementPattern::ISOTROPIC:
    case RefinementPattern::RED:
    default:
      return 8;
  }
}

size_t WedgeRefinementRule::num_children(CellFamily family, const RefinementSpec& spec) const {
  (void)spec.selector;
  return num_children(family, spec.pattern);
}

RefinedElement WedgeRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {
  if (family != CellFamily::Wedge || vertices.size() != 6) {
    throw std::invalid_argument("Invalid element for wedge refinement");
  }

  switch (spec.pattern) {
    case RefinementPattern::GREEN:
      return green_refine(vertices, spec.selector % 3u, level);
    case RefinementPattern::ISOTROPIC:
    case RefinementPattern::RED:
    default:
      return red_refine(vertices, level);
  }
}

std::vector<RefinementPattern> WedgeRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Wedge) return {};
  return {RefinementPattern::ISOTROPIC, RefinementPattern::GREEN};
}

RefinementPattern WedgeRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::ISOTROPIC;
}

RefinedElement WedgeRefinementRule::red_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {
  // Matches the unit-test expectations in test_WedgeRefinement.cpp.
  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::RED;

  refined.new_vertices.reserve(12);
  refined.new_vertex_weights.reserve(12);

  // Bottom triangle edge midpoints: 6,7,8
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[0], vertices[1])); // 6
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[1], vertices[2])); // 7
  refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[2], vertices[0])); // 8
  refined.new_vertex_weights.push_back({{2, 0.5}, {0, 0.5}});

  // Top triangle edge midpoints: 9,10,11
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[3], vertices[4])); // 9
  refined.new_vertex_weights.push_back({{3, 0.5}, {4, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[4], vertices[5])); // 10
  refined.new_vertex_weights.push_back({{4, 0.5}, {5, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[5], vertices[3])); // 11
  refined.new_vertex_weights.push_back({{5, 0.5}, {3, 0.5}});

  // Vertical edge midpoints: 12,13,14
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[0], vertices[3])); // 12
  refined.new_vertex_weights.push_back({{0, 0.5}, {3, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[1], vertices[4])); // 13
  refined.new_vertex_weights.push_back({{1, 0.5}, {4, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[2], vertices[5])); // 14
  refined.new_vertex_weights.push_back({{2, 0.5}, {5, 0.5}});

  // Mid-layer vertices: midpoints between bottom and top edge midpoints
  // 15 = mid(6,9), 16 = mid(7,10), 17 = mid(8,11)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(refined.new_vertices[0], refined.new_vertices[3])); // 15
  refined.new_vertex_weights.push_back({{0, 0.25}, {1, 0.25}, {3, 0.25}, {4, 0.25}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(refined.new_vertices[1], refined.new_vertices[4])); // 16
  refined.new_vertex_weights.push_back({{1, 0.25}, {2, 0.25}, {4, 0.25}, {5, 0.25}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(refined.new_vertices[2], refined.new_vertices[5])); // 17
  refined.new_vertex_weights.push_back({{2, 0.25}, {0, 0.25}, {5, 0.25}, {3, 0.25}});

  // Child connectivity (local indices: original 0..5, new 6..17).
  refined.child_connectivity = {
      // Bottom layer
      {0, 6, 8, 12, 15, 17},  // Child 0
      {6, 1, 7, 15, 13, 16},  // Child 1
      {8, 7, 2, 17, 16, 14},  // Child 2
      {6, 7, 8, 15, 16, 17},  // Child 3
      // Top layer
      {12, 15, 17, 3, 9, 11},   // Child 4
      {15, 13, 16, 9, 4, 10},   // Child 5
      {17, 16, 14, 11, 10, 5},  // Child 6
      {15, 16, 17, 9, 10, 11}   // Child 7
  };
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Wedge);
  return refined;
}

RefinedElement WedgeRefinementRule::green_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t base_edge,
    size_t level) const {
  // Quad-face closure refinement: split the selected quad face (base_edge) into a 2x2 grid,
  // yielding 4 wedge children. This is used by 3D RED–GREEN closure tests.
  const size_t e = base_edge % 3u;

  // Local wedge convention:
  // - bottom tri: 0,1,2
  // - top tri:    3,4,5 (0->3,1->4,2->5)
  const auto edge_endpoints = [&](size_t edge_id) -> std::array<size_t, 2> {
    switch (edge_id) {
      case 0: return {0u, 1u};
      case 1: return {1u, 2u};
      case 2: return {2u, 0u};
      default: return {0u, 1u};
    }
  };
  const std::array<size_t, 2> ab = edge_endpoints(e);
  const size_t a0 = ab[0];
  const size_t a1 = ab[1];
  const size_t c0 = 3u - a0 - a1; // remaining vertex in {0,1,2}
  const size_t a0_top = a0 + 3u;
  const size_t a1_top = a1 + 3u;
  const size_t c0_top = c0 + 3u;

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::GREEN;

  refined.new_vertices.reserve(6);
  refined.new_vertex_weights.reserve(6);

  // 6: bottom midpoint of edge (a0,a1)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[a0], vertices[a1]));
  refined.new_vertex_weights.push_back({{a0, 0.5}, {a1, 0.5}});
  // 7: top midpoint of edge (a0_top,a1_top)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[a0_top], vertices[a1_top]));
  refined.new_vertex_weights.push_back({{a0_top, 0.5}, {a1_top, 0.5}});
  // 8: vertical midpoint (a0,a0_top)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[a0], vertices[a0_top]));
  refined.new_vertex_weights.push_back({{a0, 0.5}, {a0_top, 0.5}});
  // 9: vertical midpoint (a1,a1_top)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[a1], vertices[a1_top]));
  refined.new_vertex_weights.push_back({{a1, 0.5}, {a1_top, 0.5}});
  // 10: vertical midpoint (c0,c0_top)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[c0], vertices[c0_top]));
  refined.new_vertex_weights.push_back({{c0, 0.5}, {c0_top, 0.5}});
  // 11: quad-face center (mid between bottom/top edge midpoints)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(refined.new_vertices[0], refined.new_vertices[1]));
  refined.new_vertex_weights.push_back({{a0, 0.25}, {a1, 0.25}, {a0_top, 0.25}, {a1_top, 0.25}});

  const size_t m_ab_bottom = 6;
  const size_t m_ab_top = 7;
  const size_t m_a0 = 8;
  const size_t m_a1 = 9;
  const size_t m_c = 10;
  const size_t m_center = 11;

  // 4 child wedges (2 bottom layer + 2 top layer).
  refined.child_connectivity = {
      {a0, m_ab_bottom, c0, m_a0, m_center, m_c},
      {m_ab_bottom, a1, c0, m_center, m_a1, m_c},
      {m_a0, m_center, m_c, a0_top, m_ab_top, c0_top},
      {m_center, m_a1, m_c, m_ab_top, a1_top, c0_top},
  };
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Wedge);
  return refined;
}

// ====================
// PyramidRefinementRule Implementation
// ====================

bool PyramidRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Pyramid && level < 10;
}

size_t PyramidRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Pyramid) return 0;
  switch (pattern) {
    case RefinementPattern::ANISOTROPIC:
      return 4;
    case RefinementPattern::ISOTROPIC:
    case RefinementPattern::RED:
    default:
      return 10;
  }
}

size_t PyramidRefinementRule::num_children(CellFamily family, const RefinementSpec& spec) const {
  (void)spec.selector;
  return num_children(family, spec.pattern);
}

RefinedElement PyramidRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {
  if (family != CellFamily::Pyramid || vertices.size() != 5) {
    throw std::invalid_argument("Invalid element for pyramid refinement");
  }

  switch (spec.pattern) {
    case RefinementPattern::ANISOTROPIC:
      return base_split_refine(vertices, level);
    case RefinementPattern::ISOTROPIC:
    case RefinementPattern::RED:
    default:
      return red_refine(vertices, level);
  }
}

std::vector<RefinementPattern> PyramidRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Pyramid) return {};
  return {RefinementPattern::ISOTROPIC, RefinementPattern::ANISOTROPIC};
}

RefinementPattern PyramidRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::ISOTROPIC;
}

RefinedElement PyramidRefinementRule::red_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {
  // Matches the unit-test expectations in test_PyramidRefinement.cpp.
  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::RED;

  refined.new_vertices.reserve(10);
  refined.new_vertex_weights.reserve(10);

  // Base edge midpoints: 5,6,7,8
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[0], vertices[1])); // 5
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[1], vertices[2])); // 6
  refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[2], vertices[3])); // 7
  refined.new_vertex_weights.push_back({{2, 0.5}, {3, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[3], vertices[0])); // 8
  refined.new_vertex_weights.push_back({{3, 0.5}, {0, 0.5}});

  // Lateral edge midpoints: 9,10,11,12
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[0], vertices[4])); // 9
  refined.new_vertex_weights.push_back({{0, 0.5}, {4, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[1], vertices[4])); // 10
  refined.new_vertex_weights.push_back({{1, 0.5}, {4, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[2], vertices[4])); // 11
  refined.new_vertex_weights.push_back({{2, 0.5}, {4, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[3], vertices[4])); // 12
  refined.new_vertex_weights.push_back({{3, 0.5}, {4, 0.5}});

  // Base center: 13
  const std::array<real_t, 3> base_center = RefinementUtils::face_center({vertices[0], vertices[1], vertices[2], vertices[3]});
  refined.new_vertices.push_back(base_center); // 13
  refined.new_vertex_weights.push_back({{0, 0.25}, {1, 0.25}, {2, 0.25}, {3, 0.25}});

  // Mid-height center: 14 (midpoint between base center and apex)
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(base_center, vertices[4])); // 14
  refined.new_vertex_weights.push_back({{0, 0.125}, {1, 0.125}, {2, 0.125}, {3, 0.125}, {4, 0.5}});

  // Child connectivity (local indices: original 0..4, new 5..14).
  refined.child_connectivity = {
      // Pyramids (children 0-5)
      {0, 5, 13, 8, 9},     // Child 0
      {5, 1, 6, 13, 10},    // Child 1
      {13, 6, 2, 7, 11},    // Child 2
      {8, 13, 7, 3, 12},    // Child 3
      {5, 6, 7, 8, 14},     // Child 4 (central)
      {9, 10, 11, 12, 4},   // Child 5 (apex)
      // Tetrahedra (children 6-9)
      {5, 10, 14, 9},       // Child 6
      {6, 11, 14, 10},      // Child 7
      {7, 12, 14, 11},      // Child 8
      {8, 9, 14, 12}        // Child 9
  };

  refined.child_families.clear();
  refined.child_families.reserve(refined.child_connectivity.size());
  for (size_t i = 0; i < 6; ++i) refined.child_families.push_back(CellFamily::Pyramid);
  for (size_t i = 0; i < 4; ++i) refined.child_families.push_back(CellFamily::Tetra);
  return refined;
}

RefinedElement PyramidRefinementRule::base_split_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t level) const {
  // Split only the base quad (selector-independent). Produces 4 child pyramids.
  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::ANISOTROPIC;

  refined.new_vertices.reserve(5);
  refined.new_vertex_weights.reserve(5);

  // Base edge midpoints: 5,6,7,8
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[0], vertices[1])); // 5
  refined.new_vertex_weights.push_back({{0, 0.5}, {1, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[1], vertices[2])); // 6
  refined.new_vertex_weights.push_back({{1, 0.5}, {2, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[2], vertices[3])); // 7
  refined.new_vertex_weights.push_back({{2, 0.5}, {3, 0.5}});
  refined.new_vertices.push_back(RefinementUtils::edge_midpoint(vertices[3], vertices[0])); // 8
  refined.new_vertex_weights.push_back({{3, 0.5}, {0, 0.5}});

  // Base center: 9
  refined.new_vertices.push_back(RefinementUtils::face_center({vertices[0], vertices[1], vertices[2], vertices[3]})); // 9
  refined.new_vertex_weights.push_back({{0, 0.25}, {1, 0.25}, {2, 0.25}, {3, 0.25}});

  // 4 child pyramids with shared apex 4.
  refined.child_connectivity = {
      {0, 5, 9, 8, 4},
      {5, 1, 6, 9, 4},
      {9, 6, 2, 7, 4},
      {8, 9, 7, 3, 4},
  };
  refined.child_families.assign(refined.child_connectivity.size(), CellFamily::Pyramid);
  return refined;
}

// ====================
// RefinementRulesManager Implementation
// ====================

RefinementRulesManager& RefinementRulesManager::instance() {
  static RefinementRulesManager instance;
  return instance;
}

RefinementRulesManager::RefinementRulesManager() {
  // Create and register rules
  rules_[static_cast<size_t>(CellFamily::Line)] = std::make_unique<LineRefinementRule>();
  rules_[static_cast<size_t>(CellFamily::Triangle)] = std::make_unique<TriangleRefinementRule>();
  rules_[static_cast<size_t>(CellFamily::Quad)] = std::make_unique<QuadRefinementRule>();
  rules_[static_cast<size_t>(CellFamily::Tetra)] = std::make_unique<TetrahedronRefinementRule>();
  rules_[static_cast<size_t>(CellFamily::Hex)] = std::make_unique<HexahedronRefinementRule>();
  rules_[static_cast<size_t>(CellFamily::Wedge)] = std::make_unique<WedgeRefinementRule>();
  rules_[static_cast<size_t>(CellFamily::Pyramid)] = std::make_unique<PyramidRefinementRule>();

  // Set up rule map
  for (size_t i = 0; i < rules_.size(); ++i) {
    rule_map_[i] = rules_[i].get();
  }
}

RefinementRule* RefinementRulesManager::get_rule(CellFamily family) const {
  size_t idx = static_cast<size_t>(family);
  if (idx < rule_map_.size()) {
    return rule_map_[idx];
  }
  return nullptr;
}

bool RefinementRulesManager::can_refine(CellFamily family, size_t level) const {
  auto rule = get_rule(family);
  return rule ? rule->can_refine(family, level) : false;
}

RefinedElement RefinementRulesManager::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    const RefinementSpec& spec,
    size_t level) const {

  auto rule = get_rule(family);
  if (!rule) {
    throw std::runtime_error("No refinement rule for element type");
  }

  return rule->refine(vertices, family, spec, level);
}

size_t RefinementRulesManager::num_children(CellFamily family, RefinementPattern pattern) const {
  auto rule = get_rule(family);
  return rule ? rule->num_children(family, pattern) : 0;
}

size_t RefinementRulesManager::num_children(CellFamily family, const RefinementSpec& spec) const {
  auto rule = get_rule(family);
  return rule ? rule->num_children(family, spec) : 0;
}

void RefinementRulesManager::register_rule(CellFamily family, std::unique_ptr<RefinementRule> rule) {
  size_t idx = static_cast<size_t>(family);
  if (idx < rules_.size()) {
    rules_[idx] = std::move(rule);
    rule_map_[idx] = rules_[idx].get();
  }
}

// ====================
// RefinementUtils Implementation
// ====================

std::array<real_t, 3> RefinementUtils::edge_midpoint(
    const std::array<real_t, 3>& v1,
    const std::array<real_t, 3>& v2) {

  std::array<real_t, 3> midpoint;
  for (int i = 0; i < 3; ++i) {
    midpoint[i] = static_cast<real_t>(0.5) * (v1[i] + v2[i]);
  }
  return midpoint;
}

std::array<real_t, 3> RefinementUtils::face_center(
    const std::vector<std::array<real_t, 3>>& face_vertices) {

  std::array<real_t, 3> center = {static_cast<real_t>(0.0),
                                  static_cast<real_t>(0.0),
                                  static_cast<real_t>(0.0)};
  for (const auto& v : face_vertices) {
    for (int i = 0; i < 3; ++i) {
      center[i] += v[i];
    }
  }

  if (!face_vertices.empty()) {
    for (int i = 0; i < 3; ++i) {
      center[i] /= face_vertices.size();
    }
  }

  return center;
}

std::array<real_t, 3> RefinementUtils::cell_center(
    const std::vector<std::array<real_t, 3>>& cell_vertices) {

  return face_center(cell_vertices);  // Same computation
}

size_t RefinementUtils::find_longest_edge(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family) {

  real_t max_length_sq = static_cast<real_t>(0.0);
  size_t longest_edge = 0;

  // Define edges based on element type
  std::vector<std::pair<size_t, size_t>> edges;

  switch (family) {
    case CellFamily::Triangle:
      edges = {{0,1}, {1,2}, {2,0}};
      break;
    case CellFamily::Tetra:
      edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
      break;
    default:
      return 0;
  }

  for (size_t i = 0; i < edges.size(); ++i) {
    const auto& v1 = vertices[edges[i].first];
    const auto& v2 = vertices[edges[i].second];

    real_t length_sq = static_cast<real_t>(0.0);
    for (int j = 0; j < 3; ++j) {
      const real_t diff = v2[j] - v1[j];
      length_sq += diff * diff;
    }

    if (length_sq > max_length_sq) {
      max_length_sq = length_sq;
      longest_edge = i;
    }
  }

  return longest_edge;
}

bool RefinementUtils::check_refinement_quality(
    const RefinedElement& refined,
    real_t min_quality) {

  // TODO: Implement quality checking for refined elements
  // This would compute quality metrics for each child element

  return true;  // Placeholder
}

std::vector<std::pair<size_t, size_t>> RefinementUtils::generate_edge_connectivity(
    const RefinedElement& refined) {

  std::vector<std::pair<size_t, size_t>> edges;

  // TODO: Generate edge connectivity from child elements

  return edges;
}

} // namespace svmp
