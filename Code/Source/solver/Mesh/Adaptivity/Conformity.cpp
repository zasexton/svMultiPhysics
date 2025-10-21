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

#include "Conformity.h"
#include "../MeshBase.h"
#include "../MeshFields.h"
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace svmp {

namespace {

// Helper to check if vertices form an edge
bool is_edge(const std::vector<size_t>& elem_vertices,
             size_t v1, size_t v2) {
  bool has_v1 = false;
  bool has_v2 = false;

  for (size_t v : elem_vertices) {
    if (v == v1) has_v1 = true;
    if (v == v2) has_v2 = true;
  }

  return has_v1 && has_v2;
}

// Helper to check if vertices form a face
bool contains_face(const std::vector<size_t>& elem_vertices,
                   const std::vector<size_t>& face_vertices) {
  for (size_t v : face_vertices) {
    if (std::find(elem_vertices.begin(), elem_vertices.end(), v) ==
        elem_vertices.end()) {
      return false;
    }
  }
  return true;
}

// Get refinement level of element (simplified)
size_t get_element_level(size_t elem_id) {
  // In practice, would track actual refinement levels
  // For now, use a simple heuristic based on element ID
  size_t level = 0;
  size_t id = elem_id;
  while (id > 100) {
    id /= 4; // Assume 4-to-1 refinement
    level++;
  }
  return level;
}

} // anonymous namespace

//=============================================================================
// ClosureConformityEnforcer Implementation
//=============================================================================

ClosureConformityEnforcer::ClosureConformityEnforcer(const Config& config)
    : config_(config) {}

NonConformity ClosureConformityEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {

  NonConformity non_conformity;
  non_conformity.max_level_difference = 0;

  size_t num_elements = mesh.num_elements();

  // Check each edge for conformity
  if (config_.check_edge_conformity) {
    std::set<std::pair<size_t, size_t>> processed_edges;

    for (size_t elem = 0; elem < num_elements; ++elem) {
      auto vertices = mesh.get_element_vertex_ids(elem);

      // Check all edges of this element
      for (size_t i = 0; i < vertices.size(); ++i) {
        for (size_t j = i + 1; j < vertices.size(); ++j) {
          size_t v1 = std::min(vertices[i], vertices[j]);
          size_t v2 = std::max(vertices[i], vertices[j]);

          if (processed_edges.count({v1, v2}) > 0) {
            continue;
          }
          processed_edges.insert({v1, v2});

          if (!is_edge_conforming(mesh, v1, v2, marks)) {
            non_conformity.non_conforming_edges.insert({v1, v2});

            // Find elements sharing this edge
            auto edge_elements = find_edge_elements(mesh, v1, v2);

            // Check for hanging nodes
            for (size_t edge_elem : edge_elements) {
              if (marks[edge_elem] == MarkType::REFINE) {
                // This element will create hanging nodes
                size_t level = get_element_level(edge_elem);

                // Find midpoint node (simplified)
                size_t mid_node = num_elements + elem; // Placeholder

                HangingNode hanging;
                hanging.node_id = mid_node;
                hanging.parent_entity = {v1, v2};
                hanging.on_edge = true;
                hanging.level_difference = 1;

                // Set constraint coefficients (linear interpolation)
                hanging.constraints[v1] = 0.5;
                hanging.constraints[v2] = 0.5;

                non_conformity.hanging_nodes.push_back(hanging);
              }
            }

            // Mark neighbors for closure if needed
            for (size_t edge_elem : edge_elements) {
              if (marks[edge_elem] == MarkType::NONE) {
                non_conformity.elements_needing_closure.insert(edge_elem);
              }
            }
          }
        }
      }
    }
  }

  // Check faces for conformity (3D)
  if (config_.check_face_conformity) {
    for (size_t elem = 0; elem < num_elements; ++elem) {
      auto elem_type = mesh.get_element_type(elem);

      if (elem_type == ElementType::TETRAHEDRON ||
          elem_type == ElementType::HEXAHEDRON) {
        auto vertices = mesh.get_element_vertex_ids(elem);

        // Get faces based on element type
        std::vector<std::vector<size_t>> faces;

        if (elem_type == ElementType::TETRAHEDRON) {
          // 4 triangular faces
          faces = {{vertices[0], vertices[1], vertices[2]},
                   {vertices[0], vertices[1], vertices[3]},
                   {vertices[0], vertices[2], vertices[3]},
                   {vertices[1], vertices[2], vertices[3]}};
        } else if (elem_type == ElementType::HEXAHEDRON) {
          // 6 quad faces
          faces = {{vertices[0], vertices[1], vertices[2], vertices[3]},
                   {vertices[4], vertices[5], vertices[6], vertices[7]},
                   {vertices[0], vertices[1], vertices[5], vertices[4]},
                   {vertices[2], vertices[3], vertices[7], vertices[6]},
                   {vertices[0], vertices[3], vertices[7], vertices[4]},
                   {vertices[1], vertices[2], vertices[6], vertices[5]}};
        }

        // Check each face
        for (const auto& face : faces) {
          if (!is_face_conforming(mesh, face, marks)) {
            // Sort face vertices for consistent comparison
            auto sorted_face = face;
            std::sort(sorted_face.begin(), sorted_face.end());
            non_conformity.non_conforming_faces.insert(sorted_face);

            // Find face elements and mark for closure
            auto face_elements = find_face_elements(mesh, face);
            for (size_t face_elem : face_elements) {
              if (marks[face_elem] == MarkType::NONE) {
                non_conformity.elements_needing_closure.insert(face_elem);
              }
            }
          }
        }
      }
    }
  }

  // Compute maximum level difference
  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto neighbors = mesh.get_element_neighbors(elem);
    size_t elem_level = get_element_level(elem);

    for (size_t neighbor : neighbors) {
      size_t neighbor_level = get_element_level(neighbor);
      size_t diff = (elem_level > neighbor_level) ?
                    (elem_level - neighbor_level) :
                    (neighbor_level - elem_level);

      non_conformity.max_level_difference =
          std::max(non_conformity.max_level_difference, diff);
    }
  }

  return non_conformity;
}

size_t ClosureConformityEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {

  size_t iterations = 0;
  bool changed = true;

  while (changed && iterations < config_.max_iterations) {
    changed = false;

    // Check conformity
    auto non_conformity = check_conformity(mesh, marks);

    if (non_conformity.is_conforming()) {
      break; // Mesh is conforming
    }

    // Mark elements for closure refinement
    for (size_t elem : non_conformity.elements_needing_closure) {
      if (marks[elem] == MarkType::NONE) {
        mark_for_closure(marks, elem);
        changed = true;

        // Propagate closure if needed
        if (config_.propagate_closure) {
          auto neighbors = mesh.get_element_neighbors(elem);
          for (size_t neighbor : neighbors) {
            size_t elem_level = get_element_level(elem);
            size_t neighbor_level = get_element_level(neighbor);

            if (neighbor_level < elem_level &&
                marks[neighbor] == MarkType::NONE) {
              mark_for_closure(marks, neighbor);
            }
          }
        }
      }
    }

    // Check level differences
    if (config_.max_level_difference > 0) {
      size_t num_elements = mesh.num_elements();

      for (size_t elem = 0; elem < num_elements; ++elem) {
        if (marks[elem] == MarkType::REFINE) {
          size_t elem_level = get_element_level(elem) + 1; // After refinement

          auto neighbors = mesh.get_element_neighbors(elem);
          for (size_t neighbor : neighbors) {
            size_t neighbor_level = get_element_level(neighbor);

            if (marks[neighbor] != MarkType::REFINE) {
              size_t diff = elem_level - neighbor_level;

              if (diff > config_.max_level_difference) {
                mark_for_closure(marks, neighbor);
                changed = true;
              }
            }
          }
        }
      }
    }

    iterations++;
  }

  return iterations;
}

std::map<size_t, std::map<size_t, double>>
ClosureConformityEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {

  std::map<size_t, std::map<size_t, double>> constraints;

  // Generate constraints for hanging nodes
  for (const auto& hanging : non_conformity.hanging_nodes) {
    constraints[hanging.node_id] = hanging.constraints;
  }

  return constraints;
}

bool ClosureConformityEnforcer::is_edge_conforming(
    const MeshBase& mesh,
    size_t v1, size_t v2,
    const std::vector<MarkType>& marks) const {

  // Find elements sharing this edge
  auto edge_elements = find_edge_elements(mesh, v1, v2);

  if (edge_elements.size() < 2) {
    return true; // Boundary edge
  }

  // Check if refinement marks are consistent
  bool has_refined = false;
  bool has_unrefined = false;

  for (size_t elem : edge_elements) {
    if (marks[elem] == MarkType::REFINE) {
      has_refined = true;
    } else {
      has_unrefined = true;
    }
  }

  // Non-conforming if some elements are refined and others are not
  return !(has_refined && has_unrefined);
}

bool ClosureConformityEnforcer::is_face_conforming(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices,
    const std::vector<MarkType>& marks) const {

  // Find elements sharing this face
  auto face_elements = find_face_elements(mesh, face_vertices);

  if (face_elements.size() < 2) {
    return true; // Boundary face
  }

  // Check if refinement marks are consistent
  bool has_refined = false;
  bool has_unrefined = false;

  for (size_t elem : face_elements) {
    if (marks[elem] == MarkType::REFINE) {
      has_refined = true;
    } else {
      has_unrefined = true;
    }
  }

  return !(has_refined && has_unrefined);
}

std::vector<size_t> ClosureConformityEnforcer::find_edge_elements(
    const MeshBase& mesh, size_t v1, size_t v2) const {

  std::vector<size_t> edge_elements;
  size_t num_elements = mesh.num_elements();

  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto vertices = mesh.get_element_vertex_ids(elem);

    if (is_edge(vertices, v1, v2)) {
      edge_elements.push_back(elem);
    }
  }

  return edge_elements;
}

std::vector<size_t> ClosureConformityEnforcer::find_face_elements(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices) const {

  std::vector<size_t> face_elements;
  size_t num_elements = mesh.num_elements();

  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto vertices = mesh.get_element_vertex_ids(elem);

    if (contains_face(vertices, face_vertices)) {
      face_elements.push_back(elem);
    }
  }

  return face_elements;
}

void ClosureConformityEnforcer::mark_for_closure(
    std::vector<MarkType>& marks, size_t elem_id) const {

  if (elem_id < marks.size()) {
    if (config_.use_green_closure && marks[elem_id] == MarkType::NONE) {
      // Try green refinement first
      marks[elem_id] = MarkType::REFINE_GREEN;
    } else {
      // Regular refinement
      marks[elem_id] = MarkType::REFINE;
    }
  }
}

//=============================================================================
// HangingNodeConformityEnforcer Implementation
//=============================================================================

HangingNodeConformityEnforcer::HangingNodeConformityEnforcer(const Config& config)
    : config_(config) {}

NonConformity HangingNodeConformityEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {

  NonConformity non_conformity;
  non_conformity.max_level_difference = 0;

  size_t num_elements = mesh.num_elements();

  // Find all hanging nodes
  std::set<std::pair<size_t, size_t>> processed_edges;

  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto vertices = mesh.get_element_vertex_ids(elem);

    // Check edges
    for (size_t i = 0; i < vertices.size(); ++i) {
      for (size_t j = i + 1; j < vertices.size(); ++j) {
        size_t v1 = std::min(vertices[i], vertices[j]);
        size_t v2 = std::max(vertices[i], vertices[j]);

        if (processed_edges.count({v1, v2}) > 0) {
          continue;
        }
        processed_edges.insert({v1, v2});

        // Find hanging nodes on this edge
        auto hanging = find_edge_hanging_nodes(mesh, v1, v2, marks);
        non_conformity.hanging_nodes.insert(non_conformity.hanging_nodes.end(),
                                             hanging.begin(), hanging.end());
      }
    }

    // Check faces for 3D elements
    if (mesh.get_element_type(elem) == ElementType::TETRAHEDRON ||
        mesh.get_element_type(elem) == ElementType::HEXAHEDRON) {
      // Get element faces
      std::vector<std::vector<size_t>> faces;

      if (mesh.get_element_type(elem) == ElementType::TETRAHEDRON) {
        faces = {{vertices[0], vertices[1], vertices[2]},
                 {vertices[0], vertices[1], vertices[3]},
                 {vertices[0], vertices[2], vertices[3]},
                 {vertices[1], vertices[2], vertices[3]}};
      }
      // Add hex faces if needed

      for (const auto& face : faces) {
        auto hanging = find_face_hanging_nodes(mesh, face, marks);
        non_conformity.hanging_nodes.insert(non_conformity.hanging_nodes.end(),
                                             hanging.begin(), hanging.end());
      }
    }
  }

  // Compute level differences
  for (const auto& hanging : non_conformity.hanging_nodes) {
    non_conformity.max_level_difference =
        std::max(non_conformity.max_level_difference, hanging.level_difference);
  }

  return non_conformity;
}

size_t HangingNodeConformityEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {

  // Check for excessive hanging levels
  auto non_conformity = check_conformity(mesh, marks);

  if (non_conformity.max_level_difference > config_.max_hanging_level) {
    // Need to reduce level difference
    size_t iterations = 0;

    while (non_conformity.max_level_difference > config_.max_hanging_level &&
           iterations < 10) {
      // Find elements causing excessive level difference
      for (const auto& hanging : non_conformity.hanging_nodes) {
        if (hanging.level_difference > config_.max_hanging_level) {
          // Mark coarse neighbor for refinement
          if (hanging.on_edge) {
            size_t v1 = hanging.parent_entity.first;
            size_t v2 = hanging.parent_entity.second;

            // Find coarse element on this edge
            size_t num_elements = mesh.num_elements();
            for (size_t elem = 0; elem < num_elements; ++elem) {
              if (marks[elem] != MarkType::REFINE) {
                auto vertices = mesh.get_element_vertex_ids(elem);
                if (is_edge(vertices, v1, v2)) {
                  marks[elem] = MarkType::REFINE;
                }
              }
            }
          }
        }
      }

      iterations++;
      non_conformity = check_conformity(mesh, marks);
    }

    return iterations;
  }

  return 0; // No enforcement needed - hanging nodes are allowed
}

std::map<size_t, std::map<size_t, double>>
HangingNodeConformityEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {

  std::map<size_t, std::map<size_t, double>> constraints;

  // Generate constraint equations for each hanging node
  for (const auto& hanging : non_conformity.hanging_nodes) {
    auto constraint = generate_node_constraint(mesh, hanging);
    if (!constraint.empty()) {
      constraints[hanging.node_id] = constraint;
    }
  }

  return constraints;
}

std::vector<HangingNode> HangingNodeConformityEnforcer::find_edge_hanging_nodes(
    const MeshBase& mesh,
    size_t v1, size_t v2,
    const std::vector<MarkType>& marks) const {

  std::vector<HangingNode> hanging_nodes;

  // Find elements sharing this edge
  std::vector<size_t> edge_elements;
  size_t num_elements = mesh.num_elements();

  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto vertices = mesh.get_element_vertex_ids(elem);
    if (is_edge(vertices, v1, v2)) {
      edge_elements.push_back(elem);
    }
  }

  // Check for level differences
  if (edge_elements.size() >= 2) {
    size_t min_level = SIZE_MAX;
    size_t max_level = 0;

    for (size_t elem : edge_elements) {
      size_t level = get_element_level(elem);
      if (marks[elem] == MarkType::REFINE) {
        level++; // Will be refined
      }
      min_level = std::min(min_level, level);
      max_level = std::max(max_level, level);
    }

    if (max_level > min_level) {
      // There will be a hanging node
      HangingNode hanging;
      hanging.node_id = mesh.num_nodes() + v1 * 1000 + v2; // Placeholder ID
      hanging.parent_entity = {v1, v2};
      hanging.on_edge = true;
      hanging.level_difference = max_level - min_level;

      // Linear constraint for edge midpoint
      hanging.constraints[v1] = 0.5;
      hanging.constraints[v2] = 0.5;

      hanging_nodes.push_back(hanging);
    }
  }

  return hanging_nodes;
}

std::vector<HangingNode> HangingNodeConformityEnforcer::find_face_hanging_nodes(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices,
    const std::vector<MarkType>& marks) const {

  std::vector<HangingNode> hanging_nodes;

  if (!config_.constrain_faces) {
    return hanging_nodes;
  }

  // Find elements sharing this face
  std::vector<size_t> face_elements;
  size_t num_elements = mesh.num_elements();

  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto vertices = mesh.get_element_vertex_ids(elem);
    if (contains_face(vertices, face_vertices)) {
      face_elements.push_back(elem);
    }
  }

  // Check for level differences
  if (face_elements.size() >= 2) {
    size_t min_level = SIZE_MAX;
    size_t max_level = 0;

    for (size_t elem : face_elements) {
      size_t level = get_element_level(elem);
      if (marks[elem] == MarkType::REFINE) {
        level++;
      }
      min_level = std::min(min_level, level);
      max_level = std::max(max_level, level);
    }

    if (max_level > min_level) {
      // There will be hanging nodes on face
      HangingNode hanging;
      hanging.node_id = mesh.num_nodes() + face_vertices[0] * 1000; // Placeholder
      hanging.parent_entity = {face_vertices[0], face_vertices[1]};
      hanging.on_edge = false;
      hanging.level_difference = max_level - min_level;

      // Constraint coefficients for face center
      for (size_t v : face_vertices) {
        hanging.constraints[v] = 1.0 / face_vertices.size();
      }

      hanging_nodes.push_back(hanging);
    }
  }

  return hanging_nodes;
}

std::map<size_t, double> HangingNodeConformityEnforcer::generate_node_constraint(
    const MeshBase& mesh,
    const HangingNode& node) const {

  // Return the pre-computed constraints
  return node.constraints;
}

//=============================================================================
// MinimalClosureEnforcer Implementation
//=============================================================================

MinimalClosureEnforcer::MinimalClosureEnforcer(const Config& config)
    : config_(config) {}

NonConformity MinimalClosureEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {

  // Use standard conformity check
  ClosureConformityEnforcer::Config closure_config;
  closure_config.use_green_closure = config_.prefer_green;

  ClosureConformityEnforcer closure_enforcer(closure_config);
  return closure_enforcer.check_conformity(mesh, marks);
}

size_t MinimalClosureEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {

  size_t iterations = 0;

  while (iterations < config_.max_closure_depth) {
    // Check conformity
    auto non_conformity = check_conformity(mesh, marks);

    if (non_conformity.is_conforming()) {
      break;
    }

    // Compute minimal closure
    auto closure = compute_minimal_closure(mesh, marks, non_conformity);

    if (closure.empty()) {
      break;
    }

    // Apply closure
    bool changed = false;
    for (const auto& [elem_id, pattern] : closure) {
      if (marks[elem_id] == MarkType::NONE) {
        if (pattern == RefinementPattern::GREEN) {
          marks[elem_id] = MarkType::REFINE_GREEN;
        } else if (pattern == RefinementPattern::BLUE) {
          marks[elem_id] = MarkType::REFINE_BLUE;
        } else {
          marks[elem_id] = MarkType::REFINE;
        }
        changed = true;
      }
    }

    if (!changed) {
      break;
    }

    iterations++;
  }

  return iterations;
}

std::map<size_t, std::map<size_t, double>>
MinimalClosureEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {

  // Delegate to standard constraint generation
  HangingNodeConformityEnforcer::Config hanging_config;
  HangingNodeConformityEnforcer hanging_enforcer(hanging_config);

  return hanging_enforcer.generate_constraints(mesh, non_conformity);
}

std::vector<std::pair<size_t, RefinementPattern>>
MinimalClosureEnforcer::compute_minimal_closure(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    const NonConformity& non_conformity) const {

  std::vector<std::pair<size_t, RefinementPattern>> closure;

  // Priority queue for minimal cost closure
  using CostElement = std::pair<double, std::pair<size_t, RefinementPattern>>;
  std::priority_queue<CostElement, std::vector<CostElement>,
                      std::greater<CostElement>> pq;

  // Evaluate closure options for each non-conforming element
  for (size_t elem : non_conformity.elements_needing_closure) {
    if (marks[elem] != MarkType::NONE) {
      continue;
    }

    // Evaluate different refinement patterns
    std::vector<RefinementPattern> patterns;

    if (config_.prefer_green) {
      patterns.push_back(RefinementPattern::GREEN);
    }
    if (config_.prefer_blue) {
      patterns.push_back(RefinementPattern::BLUE);
    }
    if (config_.allow_anisotropic) {
      patterns.push_back(RefinementPattern::ANISOTROPIC_X);
      patterns.push_back(RefinementPattern::ANISOTROPIC_Y);
    }
    patterns.push_back(RefinementPattern::RED); // Always include regular

    // Evaluate cost of each pattern
    for (const auto& pattern : patterns) {
      double cost = compute_closure_cost({{elem, pattern}});
      pq.push({cost, {elem, pattern}});
    }
  }

  // Select minimal cost closures
  std::set<size_t> closed_elements;

  while (!pq.empty() && closed_elements.size() < non_conformity.elements_needing_closure.size()) {
    auto [cost, elem_pattern] = pq.top();
    pq.pop();

    size_t elem = elem_pattern.first;
    if (closed_elements.count(elem) > 0) {
      continue;
    }

    closure.push_back(elem_pattern);
    closed_elements.insert(elem);
  }

  return closure;
}

double MinimalClosureEnforcer::compute_closure_cost(
    const std::vector<std::pair<size_t, RefinementPattern>>& closure) const {

  double cost = 0.0;

  for (const auto& [elem, pattern] : closure) {
    // Base refinement cost
    double elem_cost = config_.refinement_cost;

    // Pattern complexity cost
    switch (pattern) {
      case RefinementPattern::GREEN:
        elem_cost += 0.5 * config_.pattern_cost; // Green is simpler
        break;
      case RefinementPattern::BLUE:
        elem_cost += 0.7 * config_.pattern_cost;
        break;
      case RefinementPattern::RED:
        elem_cost += 1.0 * config_.pattern_cost; // Full refinement
        break;
      case RefinementPattern::ANISOTROPIC_X:
      case RefinementPattern::ANISOTROPIC_Y:
      case RefinementPattern::ANISOTROPIC_Z:
        elem_cost += 0.6 * config_.pattern_cost;
        break;
      default:
        elem_cost += config_.pattern_cost;
    }

    cost += elem_cost;
  }

  return cost;
}

//=============================================================================
// ConformityEnforcerFactory Implementation
//=============================================================================

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create(
    const AdaptivityOptions& options) {

  switch (options.conformity) {
    case ConformityType::FULL_CLOSURE:
      return create_closure();

    case ConformityType::HANGING_NODE:
      return create_hanging_node();

    case ConformityType::MINIMAL_CLOSURE:
      return create_minimal_closure();

    default:
      return create_closure();
  }
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_closure(
    const ClosureConformityEnforcer::Config& config) {
  return std::make_unique<ClosureConformityEnforcer>(config);
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_hanging_node(
    const HangingNodeConformityEnforcer::Config& config) {
  return std::make_unique<HangingNodeConformityEnforcer>(config);
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_minimal_closure(
    const MinimalClosureEnforcer::Config& config) {
  return std::make_unique<MinimalClosureEnforcer>(config);
}

//=============================================================================
// ConformityUtils Implementation
//=============================================================================

bool ConformityUtils::is_mesh_conforming(const MeshBase& mesh) {
  // Check if mesh has any hanging nodes
  auto hanging = find_hanging_nodes(mesh);
  return hanging.empty();
}

std::vector<HangingNode> ConformityUtils::find_hanging_nodes(const MeshBase& mesh) {
  std::vector<HangingNode> hanging_nodes;

  // Would implement actual hanging node detection
  // This requires tracking refinement history and node relationships

  return hanging_nodes;
}

size_t ConformityUtils::check_level_difference(
    const MeshBase& mesh,
    size_t elem1,
    size_t elem2) {

  size_t level1 = get_element_level(elem1);
  size_t level2 = get_element_level(elem2);

  return (level1 > level2) ? (level1 - level2) : (level2 - level1);
}

void ConformityUtils::apply_constraints(
    std::vector<double>& solution,
    const std::map<size_t, std::map<size_t, double>>& constraints) {

  // Apply constraint equations to solution vector
  for (const auto& [constrained_node, equation] : constraints) {
    if (constrained_node < solution.size()) {
      double value = 0.0;

      for (const auto& [master_node, weight] : equation) {
        if (master_node < solution.size()) {
          value += weight * solution[master_node];
        }
      }

      solution[constrained_node] = value;
    }
  }
}

void ConformityUtils::eliminate_constraints(
    std::vector<std::vector<double>>& matrix,
    std::vector<double>& rhs,
    const std::map<size_t, std::map<size_t, double>>& constraints) {

  // Eliminate constrained DOFs from system
  for (const auto& [constrained_dof, equation] : constraints) {
    if (constrained_dof >= matrix.size()) continue;

    // Move constraint contributions to RHS
    for (size_t i = 0; i < matrix.size(); ++i) {
      if (i == constrained_dof) continue;

      double coeff = matrix[i][constrained_dof];
      if (std::abs(coeff) < 1e-12) continue;

      for (const auto& [master_dof, weight] : equation) {
        if (master_dof < matrix[i].size()) {
          matrix[i][master_dof] += coeff * weight;
        }
      }

      matrix[i][constrained_dof] = 0.0;
    }

    // Set constraint equation in matrix
    for (size_t j = 0; j < matrix[constrained_dof].size(); ++j) {
      matrix[constrained_dof][j] = 0.0;
    }

    for (const auto& [master_dof, weight] : equation) {
      if (master_dof < matrix[constrained_dof].size()) {
        matrix[constrained_dof][master_dof] = weight;
      }
    }

    matrix[constrained_dof][constrained_dof] = 1.0;
    rhs[constrained_dof] = 0.0; // Or appropriate value
  }
}

void ConformityUtils::write_nonconformity_to_field(
    MeshFields& fields,
    const MeshBase& mesh,
    const NonConformity& non_conformity) {

  // Create fields to visualize non-conformity
  size_t num_elements = mesh.num_elements();

  // Add hanging node field
  fields.add_field("hanging_nodes", FieldType::NODAL, 1);
  auto& hanging_field = fields.get_field("hanging_nodes");
  hanging_field.values.assign(mesh.num_nodes(), 0.0);

  for (const auto& hanging : non_conformity.hanging_nodes) {
    if (hanging.node_id < hanging_field.values.size()) {
      hanging_field.values[hanging.node_id] = static_cast<double>(hanging.level_difference);
    }
  }

  // Add non-conforming elements field
  fields.add_field("conformity_closure", FieldType::ELEMENTAL, 1);
  auto& closure_field = fields.get_field("conformity_closure");
  closure_field.values.assign(num_elements, 0.0);

  for (size_t elem : non_conformity.elements_needing_closure) {
    if (elem < closure_field.values.size()) {
      closure_field.values[elem] = 1.0;
    }
  }

  // Add level difference field
  fields.add_field("level_difference", FieldType::ELEMENTAL, 1);
  auto& level_field = fields.get_field("level_difference");
  level_field.values.assign(num_elements, 0.0);

  for (size_t elem = 0; elem < num_elements; ++elem) {
    size_t max_diff = 0;
    auto neighbors = mesh.get_element_neighbors(elem);

    for (size_t neighbor : neighbors) {
      size_t diff = check_level_difference(mesh, elem, neighbor);
      max_diff = std::max(max_diff, diff);
    }

    level_field.values[elem] = static_cast<double>(max_diff);
  }
}

} // namespace svmp