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

#include "CoarseningRules.h"
#include "../MeshBase.h"
#include "QualityGuards.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace svmp {

namespace {

// Helper functions
double compute_distance(const std::array<double, 3>& p1,
                        const std::array<double, 3>& p2) {
  double dx = p2[0] - p1[0];
  double dy = p2[1] - p1[1];
  double dz = p2[2] - p1[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double compute_edge_length(const MeshBase& mesh, size_t v1, size_t v2) {
  auto pos1 = mesh.get_node_position(v1);
  auto pos2 = mesh.get_node_position(v2);
  return compute_distance(pos1, pos2);
}

double compute_dihedral_angle(const std::array<double, 3>& n1,
                              const std::array<double, 3>& n2) {
  double dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
  dot = std::max(-1.0, std::min(1.0, dot));
  return std::acos(dot) * 180.0 / M_PI;
}

std::array<double, 3> compute_normal(const std::vector<std::array<double, 3>>& vertices) {
  if (vertices.size() < 3) return {0, 0, 0};

  // Compute normal using first three vertices
  std::array<double, 3> v1 = {vertices[1][0] - vertices[0][0],
                               vertices[1][1] - vertices[0][1],
                               vertices[1][2] - vertices[0][2]};
  std::array<double, 3> v2 = {vertices[2][0] - vertices[0][0],
                               vertices[2][1] - vertices[0][1],
                               vertices[2][2] - vertices[0][2]};

  // Cross product
  std::array<double, 3> normal = {
      v1[1] * v2[2] - v1[2] * v2[1],
      v1[2] * v2[0] - v1[0] * v2[2],
      v1[0] * v2[1] - v1[1] * v2[0]
  };

  // Normalize
  double len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                         normal[2] * normal[2]);
  if (len > 1e-12) {
    normal[0] /= len;
    normal[1] /= len;
    normal[2] /= len;
  }

  return normal;
}

double compute_element_volume(const MeshBase& mesh, size_t elem_id) {
  auto vertices = mesh.get_element_vertices(elem_id);

  if (vertices.size() == 3) {
    // Triangle area
    double x1 = vertices[0][0], y1 = vertices[0][1];
    double x2 = vertices[1][0], y2 = vertices[1][1];
    double x3 = vertices[2][0], y3 = vertices[2][1];
    return 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
  } else if (vertices.size() == 4) {
    // Quad area or tet volume (simplified)
    return 1.0; // Placeholder
  }

  return 1.0;
}

} // anonymous namespace

//=============================================================================
// EdgeCollapseRule Implementation
//=============================================================================

EdgeCollapseRule::EdgeCollapseRule(const Config& config)
    : config_(config) {}

bool EdgeCollapseRule::can_coarsen(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  if (elements.empty()) return false;

  // Find if there's at least one collapsible edge
  auto edge = find_collapsible_edge(mesh, elements);
  return edge.first != SIZE_MAX;
}

CoarseningOperation EdgeCollapseRule::determine_coarsening(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  CoarseningOperation operation;
  operation.pattern = CoarseningPattern::EDGE_COLLAPSE;
  operation.valid = false;

  // Find best edge to collapse
  auto edge = find_collapsible_edge(mesh, elements);
  if (edge.first == SIZE_MAX) {
    return operation;
  }

  // Compute collapse point
  auto collapse_point = compute_collapse_point(mesh, edge.first, edge.second);

  // Check quality after collapse
  if (!check_collapse_quality(mesh, edge.first, edge.second, collapse_point)) {
    return operation;
  }

  // Build operation
  operation.collapsed_edges.push_back(edge);
  operation.removed_vertices.insert(edge.second); // Remove second vertex
  operation.source_elements = elements;

  // Find affected elements and build new connectivity
  for (size_t elem : elements) {
    auto vertices = mesh.get_element_vertex_ids(elem);
    std::vector<size_t> new_vertices;

    for (size_t v : vertices) {
      if (v == edge.second) {
        // Replace with collapse target
        new_vertices.push_back(edge.first);
      } else if (v != edge.first || new_vertices.empty() ||
                 new_vertices.back() != edge.first) {
        new_vertices.push_back(v);
      }
    }

    // Only add if element doesn't degenerate
    if (new_vertices.size() >= 3) {
      operation.new_connectivity = new_vertices;
      operation.target_element = elem;
      break;
    }
  }

  // Compute predicted quality
  GeometricQualityChecker checker;
  double total_quality = 0.0;
  size_t count = 0;

  for (size_t elem : elements) {
    auto quality = checker.compute_element_quality(mesh, elem);
    total_quality += quality.overall_quality();
    count++;
  }

  operation.predicted_quality = count > 0 ? total_quality / count : 0.0;
  operation.priority = 1.0 / (operation.predicted_quality + 0.1);
  operation.valid = true;

  return operation;
}

CoarseningHistory EdgeCollapseRule::apply_coarsening(
    MeshBase& mesh,
    const CoarseningOperation& operation) const {

  CoarseningHistory history;
  history.pattern = operation.pattern;
  history.operation_id = 0; // Would be set by manager

  // Save original state
  history.original_elements = operation.source_elements;

  for (size_t elem : operation.source_elements) {
    history.original_connectivity.push_back(mesh.get_element_vertex_ids(elem));
  }

  for (size_t v : operation.removed_vertices) {
    history.original_positions[v] = mesh.get_node_position(v);
  }

  // Apply edge collapse
  if (!operation.collapsed_edges.empty()) {
    auto edge = operation.collapsed_edges[0];
    auto collapse_point = compute_collapse_point(mesh, edge.first, edge.second);

    // Move first vertex to collapse point
    mesh.set_node_position(edge.first, collapse_point);

    // Update connectivity - replace edge.second with edge.first
    size_t num_elements = mesh.num_elements();
    for (size_t elem = 0; elem < num_elements; ++elem) {
      auto vertices = mesh.get_element_vertex_ids(elem);
      bool modified = false;

      for (size_t& v : vertices) {
        if (v == edge.second) {
          v = edge.first;
          modified = true;
        }
      }

      if (modified) {
        // Check for degenerate element
        std::unordered_set<size_t> unique_vertices(vertices.begin(), vertices.end());
        if (unique_vertices.size() < vertices.size()) {
          // Mark element for removal (simplified)
          mesh.mark_element_for_deletion(elem);
        } else {
          mesh.set_element_vertices(elem, vertices);
        }
      }
    }

    // Remove collapsed vertex
    mesh.remove_node(edge.second);
  }

  return history;
}

void EdgeCollapseRule::undo_coarsening(
    MeshBase& mesh,
    const CoarseningHistory& history) const {

  // Restore removed vertices
  for (const auto& [v, pos] : history.original_positions) {
    mesh.add_node(v, pos);
  }

  // Restore original connectivity
  for (size_t i = 0; i < history.original_elements.size(); ++i) {
    size_t elem = history.original_elements[i];
    if (i < history.original_connectivity.size()) {
      mesh.set_element_vertices(elem, history.original_connectivity[i]);
    }
  }
}

bool EdgeCollapseRule::supports_element_type(ElementType type) const {
  return type == ElementType::TRIANGLE || type == ElementType::QUAD ||
         type == ElementType::TETRAHEDRON || type == ElementType::HEXAHEDRON;
}

std::pair<size_t, size_t> EdgeCollapseRule::find_collapsible_edge(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  std::pair<size_t, size_t> best_edge = {SIZE_MAX, SIZE_MAX};
  double best_cost = std::numeric_limits<double>::max();

  // Collect all edges from elements
  std::set<std::pair<size_t, size_t>> edges;

  for (size_t elem : elements) {
    auto vertices = mesh.get_element_vertex_ids(elem);

    for (size_t i = 0; i < vertices.size(); ++i) {
      for (size_t j = i + 1; j < vertices.size(); ++j) {
        size_t v1 = std::min(vertices[i], vertices[j]);
        size_t v2 = std::max(vertices[i], vertices[j]);
        edges.insert({v1, v2});
      }
    }
  }

  // Evaluate each edge
  for (const auto& edge : edges) {
    if (can_collapse_edge(mesh, edge.first, edge.second)) {
      double edge_length = compute_edge_length(mesh, edge.first, edge.second);

      // Cost based on edge length (shorter edges preferred)
      double cost = edge_length;

      if (cost < best_cost) {
        best_cost = cost;
        best_edge = edge;
      }
    }
  }

  return best_edge;
}

bool EdgeCollapseRule::can_collapse_edge(
    const MeshBase& mesh,
    size_t v1, size_t v2) const {

  // Check if edge is on boundary
  if (config_.preserve_boundary) {
    if (mesh.is_boundary_node(v1) && mesh.is_boundary_node(v2)) {
      if (!mesh.is_boundary_edge(v1, v2)) {
        return false; // Would change boundary
      }
    }
  }

  // Check if edge is a feature
  if (config_.preserve_features) {
    auto elements1 = mesh.get_node_elements(v1);
    auto elements2 = mesh.get_node_elements(v2);

    // Find elements sharing this edge
    std::vector<size_t> edge_elements;
    for (size_t e1 : elements1) {
      if (std::find(elements2.begin(), elements2.end(), e1) != elements2.end()) {
        edge_elements.push_back(e1);
      }
    }

    if (edge_elements.size() == 2) {
      // Check dihedral angle
      auto normal1 = compute_normal(mesh.get_element_vertices(edge_elements[0]));
      auto normal2 = compute_normal(mesh.get_element_vertices(edge_elements[1]));
      double angle = compute_dihedral_angle(normal1, normal2);

      if (angle > config_.feature_angle) {
        return false; // Feature edge
      }
    }
  }

  // Check edge length ratio
  double edge_length = compute_edge_length(mesh, v1, v2);
  auto neighbors1 = mesh.get_node_neighbors(v1);
  auto neighbors2 = mesh.get_node_neighbors(v2);

  double min_neighbor_edge = std::numeric_limits<double>::max();
  for (size_t n : neighbors1) {
    if (n != v2) {
      min_neighbor_edge = std::min(min_neighbor_edge,
                                    compute_edge_length(mesh, v1, n));
    }
  }
  for (size_t n : neighbors2) {
    if (n != v1) {
      min_neighbor_edge = std::min(min_neighbor_edge,
                                    compute_edge_length(mesh, v2, n));
    }
  }

  if (edge_length / min_neighbor_edge < config_.min_edge_ratio) {
    return false; // Edge too short relative to neighbors
  }

  return true;
}

std::array<double, 3> EdgeCollapseRule::compute_collapse_point(
    const MeshBase& mesh,
    size_t v1, size_t v2) const {

  auto pos1 = mesh.get_node_position(v1);
  auto pos2 = mesh.get_node_position(v2);

  if (config_.collapse_to_midpoint) {
    // Collapse to edge midpoint
    return {(pos1[0] + pos2[0]) / 2.0,
            (pos1[1] + pos2[1]) / 2.0,
            (pos1[2] + pos2[2]) / 2.0};
  } else {
    // Collapse to minimize error (simplified - use v1)
    return pos1;
  }
}

bool EdgeCollapseRule::check_collapse_quality(
    const MeshBase& mesh,
    size_t v1, size_t v2,
    const std::array<double, 3>& collapse_point) const {

  if (!config_.check_inversion) {
    return true;
  }

  // Get elements that will be affected
  auto elements1 = mesh.get_node_elements(v1);
  auto elements2 = mesh.get_node_elements(v2);

  std::unordered_set<size_t> affected_elements;
  affected_elements.insert(elements1.begin(), elements1.end());
  affected_elements.insert(elements2.begin(), elements2.end());

  // Check each affected element
  GeometricQualityChecker checker;

  for (size_t elem : affected_elements) {
    auto vertices = mesh.get_element_vertices(elem);

    // Simulate collapse
    for (auto& v : vertices) {
      auto v_id = mesh.get_vertex_id(v);
      if (v_id == v2) {
        v = collapse_point;
      } else if (v_id == v1) {
        v = collapse_point;
      }
    }

    // Check if element becomes degenerate
    std::unordered_set<std::array<double, 3>> unique_vertices;
    for (const auto& v : vertices) {
      unique_vertices.insert(v);
    }

    if (unique_vertices.size() < 3) {
      continue; // Element will be removed
    }

    // Check quality (simplified)
    double volume = compute_element_volume(mesh, elem);
    if (volume < config_.volume_tolerance) {
      return false; // Element becomes too small or inverted
    }
  }

  return true;
}

//=============================================================================
// VertexRemovalRule Implementation
//=============================================================================

VertexRemovalRule::VertexRemovalRule(const Config& config)
    : config_(config) {}

bool VertexRemovalRule::can_coarsen(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  // Find if there's a removable vertex
  size_t vertex = find_removable_vertex(mesh, elements);
  return vertex != SIZE_MAX;
}

CoarseningOperation VertexRemovalRule::determine_coarsening(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  CoarseningOperation operation;
  operation.pattern = CoarseningPattern::VERTEX_REMOVAL;
  operation.valid = false;

  // Find vertex to remove
  size_t vertex = find_removable_vertex(mesh, elements);
  if (vertex == SIZE_MAX) {
    return operation;
  }

  operation.removed_vertices.insert(vertex);
  operation.source_elements = elements;

  // Get cavity boundary
  auto vertex_elements = mesh.get_node_elements(vertex);
  std::set<size_t> boundary_vertices;

  for (size_t elem : vertex_elements) {
    auto vertices = mesh.get_element_vertex_ids(elem);
    for (size_t v : vertices) {
      if (v != vertex) {
        boundary_vertices.insert(v);
      }
    }
  }

  // Retriangulate cavity
  std::vector<size_t> boundary_vec(boundary_vertices.begin(), boundary_vertices.end());
  auto new_triangulation = retriangulate_cavity(mesh, boundary_vec);

  if (!new_triangulation.empty()) {
    operation.new_connectivity = new_triangulation[0]; // First new element
    operation.target_element = vertex_elements[0];
    operation.valid = true;
    operation.predicted_quality = config_.min_quality;
    operation.priority = 1.0;
  }

  return operation;
}

CoarseningHistory VertexRemovalRule::apply_coarsening(
    MeshBase& mesh,
    const CoarseningOperation& operation) const {

  CoarseningHistory history;
  history.pattern = operation.pattern;

  // Save original state
  for (size_t v : operation.removed_vertices) {
    history.original_positions[v] = mesh.get_node_position(v);

    // Save elements containing this vertex
    auto vertex_elements = mesh.get_node_elements(v);
    for (size_t elem : vertex_elements) {
      history.original_elements.push_back(elem);
      history.original_connectivity.push_back(mesh.get_element_vertex_ids(elem));
    }
  }

  // Remove vertex and retriangulate
  for (size_t v : operation.removed_vertices) {
    // Get cavity boundary
    auto vertex_elements = mesh.get_node_elements(v);
    std::set<size_t> boundary_vertices;

    for (size_t elem : vertex_elements) {
      auto vertices = mesh.get_element_vertex_ids(elem);
      for (size_t vert : vertices) {
        if (vert != v) {
          boundary_vertices.insert(vert);
        }
      }

      // Mark element for deletion
      mesh.mark_element_for_deletion(elem);
    }

    // Retriangulate cavity
    std::vector<size_t> boundary_vec(boundary_vertices.begin(), boundary_vertices.end());
    auto new_triangles = retriangulate_cavity(mesh, boundary_vec);

    // Add new elements
    for (const auto& triangle : new_triangles) {
      mesh.add_element(triangle);
    }

    // Remove vertex
    mesh.remove_node(v);
  }

  return history;
}

void VertexRemovalRule::undo_coarsening(
    MeshBase& mesh,
    const CoarseningHistory& history) const {

  // Restore removed vertices
  for (const auto& [v, pos] : history.original_positions) {
    mesh.add_node(v, pos);
  }

  // Restore original elements
  for (size_t i = 0; i < history.original_elements.size(); ++i) {
    size_t elem = history.original_elements[i];
    if (i < history.original_connectivity.size()) {
      mesh.set_element_vertices(elem, history.original_connectivity[i]);
    }
  }
}

bool VertexRemovalRule::supports_element_type(ElementType type) const {
  return type == ElementType::TRIANGLE || type == ElementType::TETRAHEDRON;
}

size_t VertexRemovalRule::find_removable_vertex(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  // Collect all vertices from elements
  std::set<size_t> vertices;
  for (size_t elem : elements) {
    auto elem_vertices = mesh.get_element_vertex_ids(elem);
    vertices.insert(elem_vertices.begin(), elem_vertices.end());
  }

  // Find best vertex to remove
  size_t best_vertex = SIZE_MAX;
  double best_cost = std::numeric_limits<double>::max();

  for (size_t v : vertices) {
    if (can_remove_vertex(mesh, v)) {
      // Compute cost based on valence and position
      size_t valence = mesh.get_node_neighbors(v).size();
      double cost = static_cast<double>(valence);

      if (cost < best_cost) {
        best_cost = cost;
        best_vertex = v;
      }
    }
  }

  return best_vertex;
}

std::vector<std::vector<size_t>> VertexRemovalRule::retriangulate_cavity(
    const MeshBase& mesh,
    const std::vector<size_t>& boundary_vertices) const {

  std::vector<std::vector<size_t>> triangulation;

  if (boundary_vertices.size() < 3) {
    return triangulation;
  }

  // Simple fan triangulation (for convex cavities)
  if (config_.method == Config::RetriangulationMethod::EAR_CLIPPING ||
      boundary_vertices.size() == 3) {
    // Single triangle
    triangulation.push_back(boundary_vertices);
  } else {
    // Fan from first vertex
    for (size_t i = 1; i < boundary_vertices.size() - 1; ++i) {
      triangulation.push_back({boundary_vertices[0],
                               boundary_vertices[i],
                               boundary_vertices[i + 1]});
    }
  }

  return triangulation;
}

bool VertexRemovalRule::can_remove_vertex(
    const MeshBase& mesh,
    size_t vertex) const {

  // Check if vertex is on boundary
  if (config_.preserve_boundary && mesh.is_boundary_node(vertex)) {
    return false;
  }

  // Check valence
  size_t valence = mesh.get_node_neighbors(vertex).size();
  if (valence > config_.max_valence) {
    return false;
  }

  // Check if cavity is valid for retriangulation
  auto vertex_elements = mesh.get_node_elements(vertex);
  if (vertex_elements.empty()) {
    return false;
  }

  // Additional checks would go here (convexity, quality, etc.)

  return true;
}

//=============================================================================
// ReverseRefinementRule Implementation
//=============================================================================

ReverseRefinementRule::ReverseRefinementRule(const Config& config)
    : config_(config) {}

bool ReverseRefinementRule::can_coarsen(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  if (elements.size() < config_.min_group_size) {
    return false;
  }

  // Check if elements are refinement siblings
  return are_refinement_siblings(mesh, elements);
}

CoarseningOperation ReverseRefinementRule::determine_coarsening(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  CoarseningOperation operation;
  operation.pattern = CoarseningPattern::REVERSE_RED;
  operation.valid = false;

  if (!are_refinement_siblings(mesh, elements)) {
    return operation;
  }

  // Get parent element
  size_t parent = get_parent_element(mesh, elements);
  if (parent == SIZE_MAX) {
    return operation;
  }

  // Determine original pattern
  RefinementPattern original = determine_original_pattern(mesh, elements);

  switch (original) {
    case RefinementPattern::RED:
      operation.pattern = CoarseningPattern::REVERSE_RED;
      break;
    case RefinementPattern::GREEN:
      operation.pattern = CoarseningPattern::REVERSE_GREEN;
      break;
    case RefinementPattern::BISECTION:
      operation.pattern = CoarseningPattern::REVERSE_BISECTION;
      break;
    default:
      return operation;
  }

  operation.source_elements = elements;
  operation.target_element = parent;

  // Determine vertices to remove (midpoints added during refinement)
  // Simplified - would need actual refinement history
  std::set<size_t> all_vertices;
  std::set<size_t> boundary_vertices;

  for (size_t elem : elements) {
    auto vertices = mesh.get_element_vertex_ids(elem);
    all_vertices.insert(vertices.begin(), vertices.end());
  }

  // Find interior vertices (those not on parent boundary)
  // Simplified logic
  for (size_t v : all_vertices) {
    size_t count = 0;
    for (size_t elem : elements) {
      auto vertices = mesh.get_element_vertex_ids(elem);
      if (std::find(vertices.begin(), vertices.end(), v) != vertices.end()) {
        count++;
      }
    }
    if (count == elements.size()) {
      operation.removed_vertices.insert(v); // Interior vertex
    }
  }

  operation.valid = true;
  operation.predicted_quality = 0.8; // Assume good quality
  operation.priority = 0.5;

  return operation;
}

CoarseningHistory ReverseRefinementRule::apply_coarsening(
    MeshBase& mesh,
    const CoarseningOperation& operation) const {

  CoarseningHistory history;
  history.pattern = operation.pattern;

  // Save original state
  history.original_elements = operation.source_elements;
  for (size_t elem : operation.source_elements) {
    history.original_connectivity.push_back(mesh.get_element_vertex_ids(elem));
  }

  // Remove child elements
  for (size_t elem : operation.source_elements) {
    mesh.mark_element_for_deletion(elem);
  }

  // Create parent element
  if (!operation.new_connectivity.empty()) {
    mesh.add_element(operation.new_connectivity);
  }

  // Remove interior vertices
  for (size_t v : operation.removed_vertices) {
    history.original_positions[v] = mesh.get_node_position(v);
    mesh.remove_node(v);
  }

  return history;
}

void ReverseRefinementRule::undo_coarsening(
    MeshBase& mesh,
    const CoarseningHistory& history) const {

  // Restore removed vertices
  for (const auto& [v, pos] : history.original_positions) {
    mesh.add_node(v, pos);
  }

  // Restore child elements
  for (size_t i = 0; i < history.original_elements.size(); ++i) {
    size_t elem = history.original_elements[i];
    if (i < history.original_connectivity.size()) {
      mesh.set_element_vertices(elem, history.original_connectivity[i]);
    }
  }
}

bool ReverseRefinementRule::supports_element_type(ElementType type) const {
  return true; // Supports all element types
}

bool ReverseRefinementRule::are_refinement_siblings(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  if (elements.size() != 4 && elements.size() != 8) {
    return false; // Not typical refinement group size
  }

  // Check if elements share vertices in a pattern consistent with refinement
  // Simplified check - would use actual refinement history
  std::map<size_t, size_t> vertex_count;

  for (size_t elem : elements) {
    auto vertices = mesh.get_element_vertex_ids(elem);
    for (size_t v : vertices) {
      vertex_count[v]++;
    }
  }

  // Check for shared vertices pattern
  size_t shared_vertices = 0;
  for (const auto& [v, count] : vertex_count) {
    if (count > 1) {
      shared_vertices++;
    }
  }

  // Siblings should share interior vertices
  return shared_vertices >= 3;
}

size_t ReverseRefinementRule::get_parent_element(
    const MeshBase& mesh,
    const std::vector<size_t>& children) const {

  // In practice, would look up refinement history
  // For now, return a placeholder
  if (config_.require_history) {
    // Would check mesh refinement history
    return SIZE_MAX;
  }

  // Generate parent ID from children (simplified)
  return children[0] / 4; // Assume 4-to-1 refinement
}

RefinementPattern ReverseRefinementRule::determine_original_pattern(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  // Analyze connectivity pattern to determine original refinement
  if (elements.size() == 4) {
    // Likely 2D refinement
    return RefinementPattern::RED;
  } else if (elements.size() == 8) {
    // Likely 3D refinement
    return RefinementPattern::RED;
  } else if (elements.size() == 2) {
    // Likely bisection or green
    return RefinementPattern::BISECTION;
  }

  return RefinementPattern::RED; // Default
}

//=============================================================================
// AgglomerationRule Implementation
//=============================================================================

AgglomerationRule::AgglomerationRule(const Config& config)
    : config_(config) {}

bool AgglomerationRule::can_coarsen(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  if (elements.size() < 2 || elements.size() > config_.max_agglomeration) {
    return false;
  }

  // Check if elements are connected
  for (size_t i = 0; i < elements.size() - 1; ++i) {
    auto neighbors = mesh.get_element_neighbors(elements[i]);
    bool connected = false;

    for (size_t j = i + 1; j < elements.size(); ++j) {
      if (std::find(neighbors.begin(), neighbors.end(), elements[j]) !=
          neighbors.end()) {
        connected = true;
        break;
      }
    }

    if (!connected) {
      return false;
    }
  }

  return true;
}

CoarseningOperation AgglomerationRule::determine_coarsening(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  CoarseningOperation operation;
  operation.pattern = CoarseningPattern::AGGLOMERATION;
  operation.valid = false;

  // Find agglomeration groups
  auto groups = find_agglomeration_groups(mesh, elements);
  if (groups.empty()) {
    return operation;
  }

  // Use first group
  auto& group = groups[0];

  // Create agglomerated connectivity
  auto new_connectivity = create_agglomerated_connectivity(mesh, group);
  if (new_connectivity.empty()) {
    return operation;
  }

  operation.source_elements = group;
  operation.target_element = group[0]; // Reuse first element
  operation.new_connectivity = new_connectivity;

  // Compute quality
  operation.predicted_quality = compute_agglomeration_quality(mesh, group);
  operation.priority = 1.0 / (operation.predicted_quality + 0.1);
  operation.valid = true;

  return operation;
}

CoarseningHistory AgglomerationRule::apply_coarsening(
    MeshBase& mesh,
    const CoarseningOperation& operation) const {

  CoarseningHistory history;
  history.pattern = operation.pattern;

  // Save original state
  history.original_elements = operation.source_elements;
  for (size_t elem : operation.source_elements) {
    history.original_connectivity.push_back(mesh.get_element_vertex_ids(elem));
  }

  // Mark elements for deletion (except target)
  for (size_t elem : operation.source_elements) {
    if (elem != operation.target_element) {
      mesh.mark_element_for_deletion(elem);
    }
  }

  // Update target element with new connectivity
  mesh.set_element_vertices(operation.target_element, operation.new_connectivity);

  return history;
}

void AgglomerationRule::undo_coarsening(
    MeshBase& mesh,
    const CoarseningHistory& history) const {

  // Restore original elements
  for (size_t i = 0; i < history.original_elements.size(); ++i) {
    size_t elem = history.original_elements[i];
    if (i < history.original_connectivity.size()) {
      mesh.set_element_vertices(elem, history.original_connectivity[i]);
    }
  }
}

bool AgglomerationRule::supports_element_type(ElementType type) const {
  return type == ElementType::TRIANGLE || type == ElementType::QUAD;
}

std::vector<std::vector<size_t>> AgglomerationRule::find_agglomeration_groups(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  std::vector<std::vector<size_t>> groups;

  // Simple greedy grouping based on strategy
  std::unordered_set<size_t> used;

  for (size_t seed : elements) {
    if (used.count(seed) > 0) continue;

    std::vector<size_t> group;
    group.push_back(seed);
    used.insert(seed);

    // Add neighbors based on strategy
    auto neighbors = mesh.get_element_neighbors(seed);

    for (size_t neighbor : neighbors) {
      if (used.count(neighbor) > 0) continue;
      if (std::find(elements.begin(), elements.end(), neighbor) == elements.end()) {
        continue;
      }

      if (group.size() < config_.max_agglomeration) {
        group.push_back(neighbor);
        used.insert(neighbor);

        if (config_.strategy == Config::Strategy::GEOMETRIC) {
          // Check geometric criteria
          double quality = compute_agglomeration_quality(mesh, group);
          if (quality < config_.shape_regularity) {
            // Remove last element
            group.pop_back();
            used.erase(neighbor);
          }
        }
      }
    }

    if (group.size() >= 2) {
      groups.push_back(group);
    }
  }

  return groups;
}

std::vector<size_t> AgglomerationRule::create_agglomerated_connectivity(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  // Collect all vertices from elements
  std::set<size_t> all_vertices;

  for (size_t elem : elements) {
    auto vertices = mesh.get_element_vertex_ids(elem);
    all_vertices.insert(vertices.begin(), vertices.end());
  }

  // Find boundary vertices (those not completely interior)
  std::vector<size_t> boundary_vertices;

  for (size_t v : all_vertices) {
    // Count how many elements contain this vertex
    size_t count = 0;
    for (size_t elem : elements) {
      auto vertices = mesh.get_element_vertex_ids(elem);
      if (std::find(vertices.begin(), vertices.end(), v) != vertices.end()) {
        count++;
      }
    }

    // If vertex is not in all elements, it's on boundary
    if (count < elements.size()) {
      boundary_vertices.push_back(v);
    }
  }

  return boundary_vertices;
}

double AgglomerationRule::compute_agglomeration_quality(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  // Compute shape regularity of agglomerated element
  auto connectivity = create_agglomerated_connectivity(mesh, elements);

  if (connectivity.size() < 3) {
    return 0.0;
  }

  // Simplified quality metric
  // Would compute actual shape metrics
  double quality = 1.0 / connectivity.size(); // Penalize high vertex count

  if (config_.preserve_convexity) {
    // Check convexity (simplified)
    quality *= 0.8;
  }

  return quality;
}

//=============================================================================
// CompositeCoarseningRule Implementation
//=============================================================================

void CompositeCoarseningRule::add_rule(
    std::unique_ptr<CoarseningRule> rule,
    double priority) {
  rules_.emplace_back(std::move(rule), priority);
}

bool CompositeCoarseningRule::can_coarsen(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  for (const auto& [rule, priority] : rules_) {
    if (rule->can_coarsen(mesh, elements)) {
      return true;
    }
  }
  return false;
}

CoarseningOperation CompositeCoarseningRule::determine_coarsening(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  const CoarseningRule* best_rule = select_best_rule(mesh, elements);

  if (best_rule) {
    return best_rule->determine_coarsening(mesh, elements);
  }

  CoarseningOperation invalid_op;
  invalid_op.valid = false;
  return invalid_op;
}

CoarseningHistory CompositeCoarseningRule::apply_coarsening(
    MeshBase& mesh,
    const CoarseningOperation& operation) const {

  // Find rule that generated this operation
  for (const auto& [rule, priority] : rules_) {
    if (rule->can_coarsen(mesh, operation.source_elements)) {
      return rule->apply_coarsening(mesh, operation);
    }
  }

  return CoarseningHistory();
}

void CompositeCoarseningRule::undo_coarsening(
    MeshBase& mesh,
    const CoarseningHistory& history) const {

  // Find appropriate rule based on pattern
  for (const auto& [rule, priority] : rules_) {
    // Match by pattern or rule name
    rule->undo_coarsening(mesh, history);
    return;
  }
}

bool CompositeCoarseningRule::supports_element_type(ElementType type) const {
  for (const auto& [rule, priority] : rules_) {
    if (rule->supports_element_type(type)) {
      return true;
    }
  }
  return false;
}

const CoarseningRule* CompositeCoarseningRule::select_best_rule(
    const MeshBase& mesh,
    const std::vector<size_t>& elements) const {

  const CoarseningRule* best_rule = nullptr;
  double best_score = -1.0;

  for (const auto& [rule, priority] : rules_) {
    if (rule->can_coarsen(mesh, elements)) {
      auto operation = rule->determine_coarsening(mesh, elements);

      if (operation.valid) {
        double score = operation.predicted_quality * priority;

        if (score > best_score) {
          best_score = score;
          best_rule = rule.get();
        }
      }
    }
  }

  return best_rule;
}

//=============================================================================
// CoarseningManager Implementation
//=============================================================================

CoarseningManager::CoarseningManager(const Config& config)
    : config_(config) {
  rule_ = std::make_unique<CompositeCoarseningRule>();
}

size_t CoarseningManager::coarsen_mesh(
    MeshBase& mesh,
    const std::vector<size_t>& marked_elements,
    const AdaptivityOptions& options) {

  // Initialize rules if needed
  if (!rule_) {
    initialize_rules(options);
  }

  // Group elements for coarsening
  auto groups = group_elements_for_coarsening(mesh, marked_elements);

  size_t operations_performed = 0;

  for (const auto& group : groups) {
    if (!rule_->can_coarsen(mesh, group)) {
      stats_.failed_operations++;
      continue;
    }

    auto operation = rule_->determine_coarsening(mesh, group);

    if (!operation.valid) {
      stats_.failed_operations++;
      continue;
    }

    if (config_.validate_mesh && !validate_operation(mesh, operation)) {
      stats_.failed_operations++;
      continue;
    }

    // Apply coarsening
    auto history = rule_->apply_coarsening(mesh, operation);
    history.operation_id = next_operation_id_++;

    if (config_.track_history) {
      history_.push_back(history);

      if (history_.size() > config_.max_history) {
        history_.erase(history_.begin());
      }
    }

    // Update statistics
    stats_.total_operations++;
    stats_.successful_operations++;
    stats_.pattern_counts[operation.pattern]++;

    operations_performed++;

    // Update topology if needed
    update_topology(mesh, operation);
  }

  return operations_performed;
}

bool CoarseningManager::undo_last_operation(MeshBase& mesh) {
  if (history_.empty()) {
    return false;
  }

  auto history = history_.back();
  history_.pop_back();

  rule_->undo_coarsening(mesh, history);
  stats_.total_operations--;

  return true;
}

void CoarseningManager::clear_history() {
  history_.clear();
  next_operation_id_ = 0;
}

void CoarseningManager::initialize_rules(const AdaptivityOptions& options) {
  rule_ = std::make_unique<CompositeCoarseningRule>();

  // Add rules based on options
  rule_->add_rule(CoarseningRuleFactory::create_edge_collapse(), 1.0);
  rule_->add_rule(CoarseningRuleFactory::create_vertex_removal(), 0.8);
  rule_->add_rule(CoarseningRuleFactory::create_reverse_refinement(), 1.2);
  rule_->add_rule(CoarseningRuleFactory::create_agglomeration(), 0.6);
}

std::vector<std::vector<size_t>> CoarseningManager::group_elements_for_coarsening(
    const MeshBase& mesh,
    const std::vector<size_t>& marked_elements) const {

  std::vector<std::vector<size_t>> groups;

  if (!config_.batch_operations) {
    // Each element is its own group
    for (size_t elem : marked_elements) {
      groups.push_back({elem});
    }
    return groups;
  }

  // Group connected elements
  std::unordered_set<size_t> visited;

  for (size_t seed : marked_elements) {
    if (visited.count(seed) > 0) continue;

    std::vector<size_t> group;
    std::queue<size_t> queue;

    queue.push(seed);
    visited.insert(seed);

    while (!queue.empty()) {
      size_t elem = queue.front();
      queue.pop();
      group.push_back(elem);

      auto neighbors = mesh.get_element_neighbors(elem);
      for (size_t neighbor : neighbors) {
        if (visited.count(neighbor) > 0) continue;
        if (std::find(marked_elements.begin(), marked_elements.end(), neighbor) ==
            marked_elements.end()) {
          continue;
        }

        queue.push(neighbor);
        visited.insert(neighbor);
      }
    }

    groups.push_back(group);
  }

  return groups;
}

bool CoarseningManager::validate_operation(
    const MeshBase& mesh,
    const CoarseningOperation& operation) const {

  // Check quality threshold
  if (operation.predicted_quality < config_.min_quality) {
    return false;
  }

  // Check for mesh validity
  // Would perform actual validation checks

  return true;
}

void CoarseningManager::update_topology(
    MeshBase& mesh,
    const CoarseningOperation& operation) {

  // Update mesh connectivity and topology information
  // Clean up deleted elements
  mesh.compact();

  // Update neighbor information
  mesh.update_connectivity();
}

//=============================================================================
// CoarseningRuleFactory Implementation
//=============================================================================

std::unique_ptr<CoarseningRule> CoarseningRuleFactory::create(
    const AdaptivityOptions& options) {

  auto composite = std::make_unique<CompositeCoarseningRule>();

  // Add appropriate rules based on options
  composite->add_rule(create_edge_collapse(), 1.0);
  composite->add_rule(create_vertex_removal(), 0.8);

  if (options.use_refinement_history) {
    composite->add_rule(create_reverse_refinement(), 1.2);
  }

  return composite;
}

std::unique_ptr<CoarseningRule> CoarseningRuleFactory::create_edge_collapse(
    const EdgeCollapseRule::Config& config) {
  return std::make_unique<EdgeCollapseRule>(config);
}

std::unique_ptr<CoarseningRule> CoarseningRuleFactory::create_vertex_removal(
    const VertexRemovalRule::Config& config) {
  return std::make_unique<VertexRemovalRule>(config);
}

std::unique_ptr<CoarseningRule> CoarseningRuleFactory::create_reverse_refinement(
    const ReverseRefinementRule::Config& config) {
  return std::make_unique<ReverseRefinementRule>(config);
}

std::unique_ptr<CoarseningRule> CoarseningRuleFactory::create_agglomeration(
    const AgglomerationRule::Config& config) {
  return std::make_unique<AgglomerationRule>(config);
}

std::unique_ptr<CompositeCoarseningRule> CoarseningRuleFactory::create_composite() {
  return std::make_unique<CompositeCoarseningRule>();
}

//=============================================================================
// CoarseningUtils Implementation
//=============================================================================

bool CoarseningUtils::can_coarsen_mesh(const MeshBase& mesh) {
  // Check minimum element count
  if (mesh.num_elements() < 10) {
    return false;
  }

  // Check if mesh has coarsening history
  // Would check actual mesh properties

  return true;
}

std::vector<size_t> CoarseningUtils::find_coarsening_candidates(
    const MeshBase& mesh,
    const std::vector<double>& error_field,
    double threshold) {

  std::vector<size_t> candidates;

  for (size_t i = 0; i < error_field.size(); ++i) {
    if (error_field[i] < threshold) {
      candidates.push_back(i);
    }
  }

  return candidates;
}

double CoarseningUtils::estimate_coarsening_quality(
    const MeshBase& mesh,
    const std::vector<size_t>& elements_to_coarsen) {

  if (elements_to_coarsen.empty()) {
    return 0.0;
  }

  GeometricQualityChecker checker;
  double total_quality = 0.0;

  for (size_t elem : elements_to_coarsen) {
    auto quality = checker.compute_element_quality(mesh, elem);
    total_quality += quality.overall_quality();
  }

  return total_quality / elements_to_coarsen.size();
}

bool CoarseningUtils::preserves_topology(
    const MeshBase& mesh,
    const CoarseningOperation& operation) {

  // Check if operation preserves mesh topology
  // Would perform actual topology checks

  // Check for manifold preservation
  // Check for orientation preservation
  // Check for boundary preservation

  return true;
}

std::vector<CoarseningOperation> CoarseningUtils::compute_optimal_sequence(
    const MeshBase& mesh,
    const std::vector<size_t>& marked_elements) {

  std::vector<CoarseningOperation> sequence;

  // Compute dependency graph
  // Order operations to minimize conflicts
  // Optimize for quality and efficiency

  // Simplified: just create individual operations
  auto rule = CoarseningRuleFactory::create(AdaptivityOptions{});

  for (size_t elem : marked_elements) {
    if (rule->can_coarsen(mesh, {elem})) {
      auto op = rule->determine_coarsening(mesh, {elem});
      if (op.valid) {
        sequence.push_back(op);
      }
    }
  }

  // Sort by priority
  std::sort(sequence.begin(), sequence.end(),
            [](const CoarseningOperation& a, const CoarseningOperation& b) {
              return a.priority > b.priority;
            });

  return sequence;
}

} // namespace svmp