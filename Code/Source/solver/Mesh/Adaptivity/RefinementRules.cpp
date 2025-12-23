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
    RefinementPattern pattern,
    size_t level) const {

  if (family != CellFamily::Line || vertices.size() != 2) {
    throw std::invalid_argument("Invalid element for line refinement");
  }

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = pattern;

  // Create midpoint
  auto midpoint = RefinementUtils::edge_midpoint(vertices[0], vertices[1]);
  refined.new_vertices.push_back(midpoint);

  // Create two child lines
  // Child 0: vertex 0 to midpoint
  refined.child_connectivity.push_back({0, 2});  // 2 is the new midpoint index

  // Child 1: midpoint to vertex 1
  refined.child_connectivity.push_back({2, 1});

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
    RefinementPattern pattern,
    size_t level) const {

  if (family != CellFamily::Triangle || vertices.size() != 3) {
    throw std::invalid_argument("Invalid element for triangle refinement");
  }

  switch (pattern) {
    case RefinementPattern::RED:
      return red_refine(vertices, level);
    case RefinementPattern::GREEN:
      return green_refine(vertices, 0, level);  // Default to splitting edge 0
    case RefinementPattern::BISECTION:
      return bisect(vertices, level);
    default:
      return red_refine(vertices, level);
  }
}

std::vector<RefinementPattern> TriangleRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Triangle) return {};
  return {RefinementPattern::RED, RefinementPattern::GREEN, RefinementPattern::BISECTION};
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

  // Create 4 child triangles
  // Original vertices: 0, 1, 2
  // New vertices: 3 (m01), 4 (m12), 5 (m20)

  // Corner triangles
  refined.child_connectivity.push_back({0, 3, 5});  // Triangle at vertex 0
  refined.child_connectivity.push_back({3, 1, 4});  // Triangle at vertex 1
  refined.child_connectivity.push_back({5, 4, 2});  // Triangle at vertex 2

  // Center triangle
  refined.child_connectivity.push_back({3, 4, 5});

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
  index_t v0 = 0;
  index_t v1 = 0;
  index_t v2 = 0;

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

  // Create 2 child triangles
  // New vertex index is 3
  refined.child_connectivity.push_back({v0, 3, v2});  // First child
  refined.child_connectivity.push_back({3, v1, v2});  // Second child

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

RefinedElement QuadRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    RefinementPattern pattern,
    size_t level) const {

  if (family != CellFamily::Quad || vertices.size() != 4) {
    throw std::invalid_argument("Invalid element for quad refinement");
  }

  switch (pattern) {
    case RefinementPattern::ANISOTROPIC:
      return anisotropic_refine(vertices, 0, level);  // Default direction 0
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

  // Create 4 child quads
  // Original vertices: 0, 1, 2, 3
  // New vertices: 4 (m01), 5 (m12), 6 (m23), 7 (m30), 8 (center)

  refined.child_connectivity.push_back({0, 4, 8, 7});  // Bottom-left
  refined.child_connectivity.push_back({4, 1, 5, 8});  // Bottom-right
  refined.child_connectivity.push_back({8, 5, 2, 6});  // Top-right
  refined.child_connectivity.push_back({7, 8, 6, 3});  // Top-left

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

    // Two child quads
    refined.child_connectivity.push_back({0, 4, 5, 3});  // Left
    refined.child_connectivity.push_back({4, 1, 2, 5});  // Right
  } else {
    // Split in vertical direction
    auto m03 = RefinementUtils::edge_midpoint(vertices[0], vertices[3]);
    auto m12 = RefinementUtils::edge_midpoint(vertices[1], vertices[2]);

    refined.new_vertices.push_back(m03);  // 4
    refined.new_vertices.push_back(m12);  // 5

    // Two child quads
    refined.child_connectivity.push_back({0, 1, 5, 4});  // Bottom
    refined.child_connectivity.push_back({4, 5, 2, 3});  // Top
  }

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
    case RefinementPattern::BISECTION:
      return 2;
    default:
      return 8;
  }
}

RefinedElement TetrahedronRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    RefinementPattern pattern,
    size_t level) const {

  if (family != CellFamily::Tetra || vertices.size() != 4) {
    throw std::invalid_argument("Invalid element for tetrahedron refinement");
  }

  switch (pattern) {
    case RefinementPattern::BISECTION:
      return bisect(vertices, RefinementUtils::find_longest_edge(vertices, family), level);
    default:
      return red_refine(vertices, level);
  }
}

std::vector<RefinementPattern> TetrahedronRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Tetra) return {};
  return {RefinementPattern::RED, RefinementPattern::BISECTION};
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
  refined.child_connectivity.push_back({5, 7, 8, 9});

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
  index_t v0 = 0;
  index_t v1 = 0;
  index_t v2 = 0;
  index_t v3 = 0;
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

  // Create 2 child tets
  refined.child_connectivity.push_back({v0, 4, v2, v3});  // First child
  refined.child_connectivity.push_back({4, v1, v2, v3});  // Second child

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

RefinedElement HexahedronRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    RefinementPattern pattern,
    size_t level) const {

  if (family != CellFamily::Hex || vertices.size() != 8) {
    throw std::invalid_argument("Invalid element for hexahedron refinement");
  }

  switch (pattern) {
    case RefinementPattern::ANISOTROPIC:
      return anisotropic_refine(vertices, 0, level);
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
  for (const auto& v : edge_mids) refined.new_vertices.push_back(v);
  for (const auto& v : face_centers) refined.new_vertices.push_back(v);
  refined.new_vertices.push_back(center);

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

  return refined;
}

RefinedElement HexahedronRefinementRule::anisotropic_refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    size_t direction,
    size_t level) const {

  RefinedElement refined;
  refined.child_level = level + 1;
  refined.pattern = RefinementPattern::ANISOTROPIC;

  // Simplified anisotropic refinement - split in one direction only
  // This creates 4 child hexahedra

  // For now, implement splitting in Z direction
  // Create midpoints on vertical edges
  auto m04 = RefinementUtils::edge_midpoint(vertices[0], vertices[4]);
  auto m15 = RefinementUtils::edge_midpoint(vertices[1], vertices[5]);
  auto m26 = RefinementUtils::edge_midpoint(vertices[2], vertices[6]);
  auto m37 = RefinementUtils::edge_midpoint(vertices[3], vertices[7]);

  refined.new_vertices.push_back(m04);  // 8
  refined.new_vertices.push_back(m15);  // 9
  refined.new_vertices.push_back(m26);  // 10
  refined.new_vertices.push_back(m37);  // 11

  // Create 2 child hexahedra
  refined.child_connectivity.push_back({0, 1, 2, 3, 8, 9, 10, 11});   // Bottom half
  refined.child_connectivity.push_back({8, 9, 10, 11, 4, 5, 6, 7});   // Top half

  // Note: This is simplified - full anisotropic would create 4 children
  // by also splitting in one horizontal direction

  return refined;
}

// ====================
// Other Element Types (Wedge, Pyramid) - Placeholder
// ====================

bool WedgeRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Wedge && level < 10;
}

size_t WedgeRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Wedge) return 0;
  return 8;  // Standard wedge refinement
}

RefinedElement WedgeRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    RefinementPattern pattern,
    size_t level) const {
  // TODO: Implement wedge refinement
  throw std::runtime_error("Wedge refinement not yet implemented");
}

std::vector<RefinementPattern> WedgeRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Wedge) return {};
  return {RefinementPattern::ISOTROPIC};
}

RefinementPattern WedgeRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::ISOTROPIC;
}

bool PyramidRefinementRule::can_refine(CellFamily family, size_t level) const {
  return family == CellFamily::Pyramid && level < 10;
}

size_t PyramidRefinementRule::num_children(CellFamily family, RefinementPattern pattern) const {
  if (family != CellFamily::Pyramid) return 0;
  return 10;  // Standard pyramid refinement
}

RefinedElement PyramidRefinementRule::refine(
    const std::vector<std::array<real_t, 3>>& vertices,
    CellFamily family,
    RefinementPattern pattern,
    size_t level) const {
  // TODO: Implement pyramid refinement
  throw std::runtime_error("Pyramid refinement not yet implemented");
}

std::vector<RefinementPattern> PyramidRefinementRule::compatible_patterns(CellFamily family) const {
  if (family != CellFamily::Pyramid) return {};
  return {RefinementPattern::ISOTROPIC};
}

RefinementPattern PyramidRefinementRule::default_pattern(CellFamily family) const {
  return RefinementPattern::ISOTROPIC;
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
    RefinementPattern pattern,
    size_t level) const {

  auto rule = get_rule(family);
  if (!rule) {
    throw std::runtime_error("No refinement rule for element type");
  }

  return rule->refine(vertices, family, pattern, level);
}

size_t RefinementRulesManager::num_children(CellFamily family, RefinementPattern pattern) const {
  auto rule = get_rule(family);
  return rule ? rule->num_children(family, pattern) : 0;
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
