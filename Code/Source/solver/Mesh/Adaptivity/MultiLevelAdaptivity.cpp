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

#include "MultiLevelAdaptivity.h"
#include "../MeshBase.h"
#include "../MeshFields.h"
#include "CoarseningRules.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <numeric>

namespace svmp {

//=============================================================================
// MeshHierarchy Implementation
//=============================================================================

MeshHierarchy::MeshHierarchy(const Config& config)
    : config_(config) {}

void MeshHierarchy::build_from_fine(const MeshBase& fine_mesh) {
  levels_.clear();

  // Create finest level
  MeshLevel finest;
  finest.level = 0;
  finest.mesh = std::make_shared<MeshBase>(fine_mesh);
  finest.stats.num_elements = fine_mesh.num_elements();
  finest.stats.num_nodes = fine_mesh.num_nodes();
  levels_.push_back(finest);

  // Build coarser levels
  size_t current_level = 1;
  std::shared_ptr<MeshBase> current_mesh = finest.mesh;

  while (current_level < config_.max_levels) {
    std::shared_ptr<MeshBase> coarse_mesh;

    // Choose coarsening method
    switch (config_.build_method) {
      case Config::BuildMethod::GEOMETRIC:
        coarse_mesh = geometric_coarsen(*current_mesh);
        break;
      case Config::BuildMethod::ALGEBRAIC:
        coarse_mesh = algebraic_coarsen(*current_mesh);
        break;
      case Config::BuildMethod::AGGLOMERATION:
        coarse_mesh = agglomeration_coarsen(*current_mesh);
        break;
      case Config::BuildMethod::NESTED:
        // Nested refinement - reverse of refinement
        coarse_mesh = geometric_coarsen(*current_mesh);
        break;
    }

    // Check if we have enough elements to continue
    if (!coarse_mesh || coarse_mesh->num_elements() < config_.min_elements) {
      break;
    }

    // Create level
    MeshLevel level;
    level.level = current_level;
    level.mesh = coarse_mesh;
    level.stats.num_elements = coarse_mesh->num_elements();
    level.stats.num_nodes = coarse_mesh->num_nodes();

    // Build transfer operators to previous (finer) level
    build_transfer_operators(current_level, current_level - 1);

    levels_.push_back(level);

    current_mesh = coarse_mesh;
    current_level++;
  }

  // Reverse order so level 0 is coarsest
  std::reverse(levels_.begin(), levels_.end());
  for (size_t i = 0; i < levels_.size(); ++i) {
    levels_[i].level = i;
  }
}

void MeshHierarchy::build_from_coarse(const MeshBase& coarse_mesh) {
  levels_.clear();

  // Create coarsest level
  MeshLevel coarsest;
  coarsest.level = 0;
  coarsest.mesh = std::make_shared<MeshBase>(coarse_mesh);
  coarsest.stats.num_elements = coarse_mesh.num_elements();
  coarsest.stats.num_nodes = coarse_mesh.num_nodes();
  levels_.push_back(coarsest);

  // Build finer levels through refinement
  // This would use RefinementRules to refine uniformly or semi-uniformly
}

void MeshHierarchy::add_level(const MeshLevel& level) {
  levels_.push_back(level);

  // Sort by level
  std::sort(levels_.begin(), levels_.end(),
            [](const MeshLevel& a, const MeshLevel& b) {
              return a.level < b.level;
            });
}

void MeshHierarchy::transfer_field(size_t from_level, size_t to_level,
                                   const std::vector<double>& from_field,
                                   std::vector<double>& to_field) const {
  if (from_level < to_level) {
    // Prolongate (coarse to fine)
    prolongate(from_level, to_level, from_field, to_field);
  } else if (from_level > to_level) {
    // Restrict (fine to coarse)
    restrict(from_level, to_level, from_field, to_field);
  } else {
    // Same level - direct copy
    to_field = from_field;
  }
}

void MeshHierarchy::prolongate(size_t coarse_level, size_t fine_level,
                               const std::vector<double>& coarse_field,
                               std::vector<double>& fine_field) const {
  if (coarse_level >= fine_level) return;

  // Multi-level prolongation if needed
  std::vector<double> current_field = coarse_field;

  for (size_t level = coarse_level; level < fine_level; ++level) {
    const auto& operators = levels_[level + 1].operators;

    if (!operators.prolongation.empty()) {
      std::vector<double> next_field(operators.prolongation.size(), 0.0);

      // Apply prolongation operator: fine = P * coarse
      for (size_t i = 0; i < operators.prolongation.size(); ++i) {
        for (size_t j = 0; j < operators.prolongation[i].size(); ++j) {
          if (j < current_field.size()) {
            next_field[i] += operators.prolongation[i][j] * current_field[j];
          }
        }
      }

      current_field = next_field;
    } else {
      // Simple linear interpolation fallback
      const auto& coarse_mesh = levels_[level].mesh;
      const auto& fine_mesh = levels_[level + 1].mesh;

      std::vector<double> next_field(fine_mesh->num_nodes(), 0.0);

      // For each fine node, find coarse parent and interpolate
      for (size_t fine_node = 0; fine_node < fine_mesh->num_nodes(); ++fine_node) {
        auto fine_pos = fine_mesh->get_node_position(fine_node);

        // Find containing coarse element
        for (size_t coarse_elem = 0; coarse_elem < coarse_mesh->num_elements(); ++coarse_elem) {
          auto coarse_vertices = coarse_mesh->get_element_vertex_ids(coarse_elem);

          // Check if point is in element (simplified)
          bool inside = true; // Would need proper containment test

          if (inside) {
            // Linear interpolation from coarse vertices
            double value = 0.0;
            double weight_sum = 0.0;

            for (size_t cv : coarse_vertices) {
              if (cv < current_field.size()) {
                auto coarse_pos = coarse_mesh->get_node_position(cv);

                // Distance-based weight
                double dx = fine_pos[0] - coarse_pos[0];
                double dy = fine_pos[1] - coarse_pos[1];
                double dz = fine_pos[2] - coarse_pos[2];
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                double weight = 1.0 / (dist + 1e-12);

                value += weight * current_field[cv];
                weight_sum += weight;
              }
            }

            if (weight_sum > 0) {
              next_field[fine_node] = value / weight_sum;
            }
            break;
          }
        }
      }

      current_field = next_field;
    }
  }

  fine_field = current_field;
}

void MeshHierarchy::restrict(size_t fine_level, size_t coarse_level,
                             const std::vector<double>& fine_field,
                             std::vector<double>& coarse_field) const {
  if (fine_level <= coarse_level) return;

  // Multi-level restriction if needed
  std::vector<double> current_field = fine_field;

  for (size_t level = fine_level; level > coarse_level; --level) {
    const auto& operators = levels_[level].operators;

    if (!operators.restriction.empty()) {
      std::vector<double> prev_field(operators.restriction.size(), 0.0);

      // Apply restriction operator: coarse = R * fine
      for (size_t i = 0; i < operators.restriction.size(); ++i) {
        for (size_t j = 0; j < operators.restriction[i].size(); ++j) {
          if (j < current_field.size()) {
            prev_field[i] += operators.restriction[i][j] * current_field[j];
          }
        }
      }

      current_field = prev_field;
    } else {
      // Simple averaging fallback
      const auto& fine_mesh = levels_[level].mesh;
      const auto& coarse_mesh = levels_[level - 1].mesh;

      std::vector<double> prev_field(coarse_mesh->num_nodes(), 0.0);
      std::vector<size_t> counts(coarse_mesh->num_nodes(), 0);

      // For each fine node, map to coarse parent
      const auto& coarsening_map = levels_[level].coarsening_map;

      for (const auto& [fine_node, coarse_node] : coarsening_map) {
        if (fine_node < current_field.size() && coarse_node < prev_field.size()) {
          prev_field[coarse_node] += current_field[fine_node];
          counts[coarse_node]++;
        }
      }

      // Average values
      for (size_t i = 0; i < prev_field.size(); ++i) {
        if (counts[i] > 0) {
          prev_field[i] /= counts[i];
        }
      }

      current_field = prev_field;
    }
  }

  coarse_field = current_field;
}

void MeshHierarchy::build_transfer_operators(size_t coarse_level, size_t fine_level) {
  if (coarse_level >= levels_.size() || fine_level >= levels_.size()) return;

  auto& operators = levels_[fine_level].operators;
  const auto& coarse_mesh = levels_[coarse_level].mesh;
  const auto& fine_mesh = levels_[fine_level].mesh;

  size_t num_fine = fine_mesh->num_nodes();
  size_t num_coarse = coarse_mesh->num_nodes();

  // Build prolongation operator (coarse to fine)
  operators.prolongation.resize(num_fine);

  for (size_t fine_node = 0; fine_node < num_fine; ++fine_node) {
    // Find parent coarse nodes and compute interpolation weights
    // This is a simplified version - real implementation would use
    // proper parent-child relationships

    // For now, store identity-like mapping
    if (fine_node < num_coarse) {
      operators.prolongation[fine_node].push_back(1.0);
    } else {
      // Interpolate from multiple coarse nodes
      // Simplified: average of two parents
      size_t parent1 = fine_node % num_coarse;
      size_t parent2 = (fine_node + 1) % num_coarse;
      operators.prolongation[fine_node] = {0.5, 0.5};
    }
  }

  // Build restriction operator (fine to coarse)
  // Often R = c * P^T for some constant c
  operators.restriction.resize(num_coarse);

  for (size_t coarse_node = 0; coarse_node < num_coarse; ++coarse_node) {
    // Find all fine nodes that have this as parent
    std::vector<double> weights;

    for (size_t fine_node = 0; fine_node < num_fine; ++fine_node) {
      // Check if coarse_node is parent of fine_node
      // Simplified logic
      if (fine_node == coarse_node) {
        weights.push_back(1.0);
      }
    }

    if (!weights.empty()) {
      double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
      for (double& w : weights) {
        w /= sum;
      }
      operators.restriction[coarse_node] = weights;
    }
  }
}

std::shared_ptr<MeshBase> MeshHierarchy::geometric_coarsen(const MeshBase& mesh) {
  // Geometric coarsening: remove every other vertex/element
  // This is simplified - real implementation would use CoarseningRules

  auto coarse_mesh = std::make_shared<MeshBase>();

  // Coarsen by factor of 2 in each direction
  size_t target_elements = mesh.num_elements() / 4;

  if (target_elements < config_.min_elements) {
    return nullptr;
  }

  // Use edge collapse or vertex removal
  CoarseningManager coarsener;
  std::vector<size_t> marked_elements;

  // Mark every other element for coarsening
  for (size_t i = 0; i < mesh.num_elements(); i += 2) {
    marked_elements.push_back(i);
  }

  // Apply coarsening (simplified - would use actual coarsening manager)
  *coarse_mesh = mesh; // Start with copy

  return coarse_mesh;
}

std::shared_ptr<MeshBase> MeshHierarchy::algebraic_coarsen(const MeshBase& mesh) {
  // Algebraic coarsening based on connectivity strength
  auto coarse_mesh = std::make_shared<MeshBase>();

  // Build strength matrix
  size_t num_nodes = mesh.num_nodes();
  std::vector<std::vector<double>> strength(num_nodes);

  for (size_t i = 0; i < num_nodes; ++i) {
    auto neighbors = mesh.get_node_neighbors(i);
    strength[i].resize(num_nodes, 0.0);

    for (size_t j : neighbors) {
      // Compute connection strength (simplified)
      strength[i][j] = 1.0;
    }
  }

  // Use Ruge-Stuben coarsening or similar
  // Select coarse nodes (C-points) and fine nodes (F-points)
  std::vector<bool> is_coarse(num_nodes, false);

  // Greedy maximal independent set
  std::vector<int> lambda(num_nodes, 0); // Number of strong connections

  for (size_t i = 0; i < num_nodes; ++i) {
    for (size_t j = 0; j < num_nodes; ++j) {
      if (strength[i][j] > 0) {
        lambda[i]++;
      }
    }
  }

  while (true) {
    // Find node with maximum lambda
    int max_lambda = *std::max_element(lambda.begin(), lambda.end());
    if (max_lambda <= 0) break;

    size_t max_node = std::distance(lambda.begin(),
                                     std::max_element(lambda.begin(), lambda.end()));

    // Make it a C-point
    is_coarse[max_node] = true;
    lambda[max_node] = -1;

    // Update lambda for neighbors
    for (size_t j = 0; j < num_nodes; ++j) {
      if (strength[max_node][j] > 0 && lambda[j] >= 0) {
        lambda[j]--;
      }
    }
  }

  // Build coarse mesh from C-points
  // Simplified - would build actual mesh structure
  *coarse_mesh = mesh;

  return coarse_mesh;
}

std::shared_ptr<MeshBase> MeshHierarchy::agglomeration_coarsen(const MeshBase& mesh) {
  // Element agglomeration coarsening
  auto coarse_mesh = std::make_shared<MeshBase>();

  // Use agglomeration from CoarseningRules
  AgglomerationRule::Config config;
  config.max_agglomeration = 4;

  AgglomerationRule rule(config);

  // Find agglomeration groups
  std::vector<bool> visited(mesh.num_elements(), false);
  std::vector<std::vector<size_t>> groups;

  for (size_t seed = 0; seed < mesh.num_elements(); ++seed) {
    if (visited[seed]) continue;

    std::vector<size_t> group;
    std::queue<size_t> queue;

    queue.push(seed);
    visited[seed] = true;

    while (!queue.empty() && group.size() < config.max_agglomeration) {
      size_t elem = queue.front();
      queue.pop();
      group.push_back(elem);

      auto neighbors = mesh.get_element_neighbors(elem);
      for (size_t neighbor : neighbors) {
        if (!visited[neighbor] && group.size() < config.max_agglomeration) {
          queue.push(neighbor);
          visited[neighbor] = true;
        }
      }
    }

    groups.push_back(group);
  }

  // Build coarse mesh from groups
  // Each group becomes one coarse element
  *coarse_mesh = mesh; // Simplified

  return coarse_mesh;
}

//=============================================================================
// MultiGridErrorEstimator Implementation
//=============================================================================

MultiGridErrorEstimator::MultiGridErrorEstimator(const Config& config)
    : config_(config) {}

std::vector<double> MultiGridErrorEstimator::estimate_error(
    const MeshHierarchy& hierarchy,
    size_t level) const {

  switch (config_.strategy) {
    case Config::Strategy::RICHARDSON:
      return richardson_error(hierarchy, level);
    case Config::Strategy::HIERARCHICAL:
      return hierarchical_error(hierarchy, level);
    case Config::Strategy::CASCADIC:
      return richardson_error(hierarchy, level); // Simplified
    case Config::Strategy::GRADIENT_BASED:
      return hierarchical_error(hierarchy, level); // Simplified
    default:
      return std::vector<double>(
          hierarchy.get_level(level).mesh->num_elements(), 0.0);
  }
}

std::vector<double> MultiGridErrorEstimator::richardson_error(
    const MeshHierarchy& hierarchy,
    size_t level) const {

  const auto& fine_mesh = hierarchy.get_level(level).mesh;
  size_t num_elements = fine_mesh->num_elements();

  if (level == 0) {
    // Coarsest level - no coarser solution available
    return std::vector<double>(num_elements, 0.0);
  }

  // Get solutions on fine and coarse levels
  const auto& fine_fields = hierarchy.get_level(level).fields;
  const auto& coarse_fields = hierarchy.get_level(level - 1).fields;

  if (!fine_fields || !coarse_fields) {
    return std::vector<double>(num_elements, 0.0);
  }

  // Get first field
  const auto& fine_data = fine_fields->get_fields().begin()->second.values;
  const auto& coarse_data = coarse_fields->get_fields().begin()->second.values;

  // Prolongate coarse solution to fine mesh
  std::vector<double> prolonged_coarse;
  hierarchy.prolongate(level - 1, level, coarse_data, prolonged_coarse);

  // Compute error using Richardson extrapolation
  std::vector<double> error(num_elements, 0.0);

  // For nodal fields
  if (fine_data.size() == fine_mesh->num_nodes()) {
    for (size_t elem = 0; elem < num_elements; ++elem) {
      auto vertices = fine_mesh->get_element_vertex_ids(elem);

      double elem_error = 0.0;
      for (size_t v : vertices) {
        if (v < fine_data.size() && v < prolonged_coarse.size()) {
          double local_error = fine_data[v] - prolonged_coarse[v];
          elem_error += local_error * local_error;
        }
      }

      error[elem] = std::sqrt(elem_error / vertices.size());

      // Richardson extrapolation factor
      double h_ratio = config_.coarsening_ratio;
      double order = config_.extrapolation_order;
      error[elem] *= std::pow(h_ratio, order) / (std::pow(h_ratio, order) - 1.0);
    }
  }

  return error;
}

std::vector<double> MultiGridErrorEstimator::hierarchical_error(
    const MeshHierarchy& hierarchy,
    size_t level) const {

  const auto& mesh = hierarchy.get_level(level).mesh;
  size_t num_elements = mesh->num_elements();

  // Hierarchical basis error estimation
  // Error is difference between hierarchical basis contributions
  std::vector<double> error(num_elements, 0.0);

  if (level == 0) {
    return error;
  }

  const auto& fine_fields = hierarchy.get_level(level).fields;
  if (!fine_fields) {
    return error;
  }

  // Compute hierarchical surplus
  const auto& fine_data = fine_fields->get_fields().begin()->second.values;
  std::vector<double> coarse_data;

  hierarchy.restrict(level, level - 1, fine_data, coarse_data);

  std::vector<double> prolonged;
  hierarchy.prolongate(level - 1, level, coarse_data, prolonged);

  // Hierarchical surplus = fine - prolongated coarse
  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto vertices = mesh->get_element_vertex_ids(elem);

    double surplus = 0.0;
    for (size_t v : vertices) {
      if (v < fine_data.size() && v < prolonged.size()) {
        double diff = fine_data[v] - prolonged[v];
        surplus += diff * diff;
      }
    }

    error[elem] = std::sqrt(surplus / vertices.size());
  }

  return error;
}

std::vector<double> MultiGridErrorEstimator::extrapolate_solution(
    const std::vector<double>& coarse,
    const std::vector<double>& fine,
    double ratio) const {

  std::vector<double> extrapolated(fine.size());

  double order = config_.extrapolation_order;
  double factor = std::pow(ratio, order) / (std::pow(ratio, order) - 1.0);

  for (size_t i = 0; i < fine.size(); ++i) {
    double coarse_val = (i < coarse.size()) ? coarse[i] : 0.0;
    extrapolated[i] = fine[i] + factor * (fine[i] - coarse_val);
  }

  return extrapolated;
}

void MultiGridErrorEstimator::smooth_error(std::vector<double>& error,
                                          const MeshBase& mesh) const {

  for (size_t iter = 0; iter < config_.smoothing_iterations; ++iter) {
    std::vector<double> smoothed = error;

    for (size_t elem = 0; elem < error.size(); ++elem) {
      auto neighbors = mesh.get_element_neighbors(elem);

      if (neighbors.empty()) continue;

      double avg = error[elem];
      for (size_t neighbor : neighbors) {
        if (neighbor < error.size()) {
          avg += error[neighbor];
        }
      }

      smoothed[elem] = avg / (neighbors.size() + 1);
    }

    error = smoothed;
  }
}

//=============================================================================
// MultiLevelMarker Implementation
//=============================================================================

MultiLevelMarker::MultiLevelMarker(const Config& config)
    : config_(config) {}

std::vector<std::vector<MarkType>> MultiLevelMarker::mark_hierarchy(
    const MeshHierarchy& hierarchy,
    const std::vector<std::vector<double>>& errors) const {

  size_t num_levels = hierarchy.num_levels();
  std::vector<std::vector<MarkType>> marks(num_levels);

  // Mark each level independently first
  for (size_t level = 0; level < num_levels; ++level) {
    if (level >= errors.size()) continue;

    double threshold = (level < config_.level_thresholds.size()) ?
                       config_.level_thresholds[level] : 0.5;

    marks[level] = mark_level(hierarchy.get_level(level), errors[level], threshold);
  }

  // Propagate marks between levels
  propagate_marks(marks, hierarchy);

  return marks;
}

void MultiLevelMarker::propagate_marks(
    std::vector<std::vector<MarkType>>& marks,
    const MeshHierarchy& hierarchy) const {

  switch (config_.propagation) {
    case Config::Propagation::TOP_DOWN:
      propagate_down(marks, hierarchy);
      break;
    case Config::Propagation::BOTTOM_UP:
      propagate_up(marks, hierarchy);
      break;
    case Config::Propagation::SYNCHRONIZED:
      propagate_down(marks, hierarchy);
      propagate_up(marks, hierarchy);
      break;
  }
}

std::vector<MarkType> MultiLevelMarker::mark_level(
    const MeshLevel& level,
    const std::vector<double>& error,
    double threshold) const {

  size_t num_elements = level.mesh->num_elements();
  std::vector<MarkType> marks(num_elements, MarkType::NONE);

  double max_error = *std::max_element(error.begin(), error.end());
  double mark_threshold = max_error * threshold;

  for (size_t elem = 0; elem < num_elements; ++elem) {
    if (elem < error.size() && error[elem] > mark_threshold) {
      marks[elem] = MarkType::REFINE;
    }
  }

  return marks;
}

void MultiLevelMarker::propagate_down(
    std::vector<std::vector<MarkType>>& marks,
    const MeshHierarchy& hierarchy) const {

  // Propagate from fine to coarse
  for (size_t level = hierarchy.num_levels() - 1; level > 0; --level) {
    const auto& refinement_map = hierarchy.get_level(level - 1).refinement_map;

    if (config_.mark_parent_with_children) {
      // If any child is marked, mark parent
      for (const auto& [parent, children] : refinement_map) {
        bool any_child_marked = false;

        for (size_t child : children) {
          if (child < marks[level].size() &&
              marks[level][child] == MarkType::REFINE) {
            any_child_marked = true;
            break;
          }
        }

        if (any_child_marked && parent < marks[level - 1].size()) {
          marks[level - 1][parent] = MarkType::REFINE;
        }
      }
    }
  }
}

void MultiLevelMarker::propagate_up(
    std::vector<std::vector<MarkType>>& marks,
    const MeshHierarchy& hierarchy) const {

  // Propagate from coarse to fine
  for (size_t level = 0; level < hierarchy.num_levels() - 1; ++level) {
    const auto& refinement_map = hierarchy.get_level(level).refinement_map;

    if (config_.mark_children_with_parent) {
      // If parent is marked, mark all children
      for (const auto& [parent, children] : refinement_map) {
        if (parent < marks[level].size() &&
            marks[level][parent] == MarkType::REFINE) {

          for (size_t child : children) {
            if (child < marks[level + 1].size()) {
              marks[level + 1][child] = MarkType::REFINE;
            }
          }
        }
      }
    }
  }
}

//=============================================================================
// Simplified implementations for remaining classes
//=============================================================================

// FullMultiGridAdaptivity
FullMultiGridAdaptivity::FullMultiGridAdaptivity(const Config& config)
    : config_(config) {}

void FullMultiGridAdaptivity::fmg_cycle(MeshHierarchy& hierarchy) {
  // Start from coarsest level
  coarse_solve(hierarchy.get_coarsest());

  // V-cycles on each level
  for (size_t level = 1; level < hierarchy.num_levels(); ++level) {
    // Prolongate solution from coarser level
    std::vector<double> coarse_solution;
    std::vector<double> fine_solution;

    hierarchy.prolongate(level - 1, level, coarse_solution, fine_solution);

    // Perform V-cycle
    v_cycle(hierarchy, level);
  }
}

void FullMultiGridAdaptivity::v_cycle(MeshHierarchy& hierarchy, size_t level) {
  if (level == 0) {
    coarse_solve(hierarchy.get_level(0));
    return;
  }

  // Pre-smoothing
  smooth(hierarchy.get_level(level), config_.pre_smooth);

  // Compute residual
  auto residual = compute_residual(hierarchy.get_level(level));

  // Restrict to coarser level
  std::vector<double> coarse_residual;
  hierarchy.restrict(level, level - 1, residual, coarse_residual);

  // Recursive call
  v_cycle(hierarchy, level - 1);

  // Prolongate correction
  std::vector<double> correction;
  std::vector<double> fine_correction;
  hierarchy.prolongate(level - 1, level, correction, fine_correction);

  // Apply correction
  apply_correction(hierarchy.get_level(level), fine_correction);

  // Post-smoothing
  smooth(hierarchy.get_level(level), config_.post_smooth);
}

void FullMultiGridAdaptivity::w_cycle(MeshHierarchy& hierarchy, size_t level) {
  // Similar to V-cycle but with two recursive calls
  if (level == 0) {
    coarse_solve(hierarchy.get_level(0));
    return;
  }

  smooth(hierarchy.get_level(level), config_.pre_smooth);

  // Two recursive calls for W-cycle
  w_cycle(hierarchy, level - 1);
  w_cycle(hierarchy, level - 1);

  smooth(hierarchy.get_level(level), config_.post_smooth);
}

// OctreeAdaptivity
OctreeAdaptivity::OctreeAdaptivity(const Config& config)
    : config_(config) {
  root_ = std::make_unique<OctreeNode>();
}

void OctreeAdaptivity::build_tree(const MeshBase& mesh) {
  // Compute root bounds
  root_->bounds = compute_bounds(mesh, {});
  root_->level = 0;

  // Build all elements into root
  std::vector<size_t> all_elements(mesh.num_elements());
  std::iota(all_elements.begin(), all_elements.end(), 0);

  build_node(root_.get(), mesh, all_elements, 0);
}

void OctreeAdaptivity::build_node(OctreeNode* node, const MeshBase& mesh,
                                  const std::vector<size_t>& elements, size_t depth) {
  node->elements = elements;

  if (elements.size() <= config_.max_elements || depth >= config_.max_depth) {
    node->is_leaf = true;
    return;
  }

  // Split node
  split_node(node);

  // Distribute elements to children
  for (auto& child : node->children) {
    std::vector<size_t> child_elements;

    for (size_t elem : elements) {
      // Check if element is in child bounds
      auto centroid = mesh.get_element_centroid(elem);

      bool in_child = true;
      for (int dim = 0; dim < 3; ++dim) {
        if (centroid[dim] < child->bounds[dim] ||
            centroid[dim] > child->bounds[dim + 3]) {
          in_child = false;
          break;
        }
      }

      if (in_child) {
        child_elements.push_back(elem);
      }
    }

    build_node(child.get(), mesh, child_elements, depth + 1);
  }

  node->is_leaf = false;
}

// SpaceTimeAdaptivity
SpaceTimeAdaptivity::SpaceTimeAdaptivity(const Config& config)
    : config_(config) {}

void SpaceTimeAdaptivity::adapt(MeshBase& mesh,
                                double& time_step,
                                const std::vector<double>& solution,
                                double current_time) {

  // Estimate space-time error
  std::vector<double> previous_solution = solution; // Would get from history
  auto error = estimate_spacetime_error(mesh, solution, previous_solution, time_step);

  // Adapt space if needed
  if (current_time > 0 && static_cast<size_t>(current_time / time_step) % config_.spatial_frequency == 0) {
    adapt_space(mesh, error);
  }

  // Adapt time step
  if (config_.adapt_time_step) {
    adapt_time(time_step, error);
  }
}

std::vector<double> SpaceTimeAdaptivity::estimate_spacetime_error(
    const MeshBase& mesh,
    const std::vector<double>& solution,
    const std::vector<double>& previous_solution,
    double time_step) const {

  size_t num_elements = mesh.num_elements();
  std::vector<double> error(num_elements, 0.0);

  // Spatial error contribution (simplified)
  for (size_t elem = 0; elem < num_elements; ++elem) {
    auto vertices = mesh.get_element_vertex_ids(elem);

    double spatial_error = 0.0;
    for (size_t v : vertices) {
      if (v < solution.size()) {
        // Gradient-based spatial error (simplified)
        auto neighbors = mesh.get_node_neighbors(v);
        for (size_t neighbor : neighbors) {
          if (neighbor < solution.size()) {
            double gradient = std::abs(solution[v] - solution[neighbor]);
            spatial_error += gradient * gradient;
          }
        }
      }
    }

    // Temporal error contribution
    double temporal_error = 0.0;
    for (size_t v : vertices) {
      if (v < solution.size() && v < previous_solution.size()) {
        double time_deriv = (solution[v] - previous_solution[v]) / time_step;
        temporal_error += time_deriv * time_deriv;
      }
    }

    error[elem] = std::sqrt(spatial_error + temporal_error);
  }

  return error;
}

double SpaceTimeAdaptivity::compute_time_step(const MeshBase& mesh,
                                              const std::vector<double>& solution,
                                              const std::vector<double>& error) const {

  // CFL-based time step computation
  double min_h = std::numeric_limits<double>::max();

  for (size_t elem = 0; elem < mesh.num_elements(); ++elem) {
    double h = mesh.get_element_diameter(elem);
    min_h = std::min(min_h, h);
  }

  // Max velocity (simplified)
  double max_vel = 1.0;

  double dt = config_.cfl_number * min_h / max_vel;

  // Apply bounds
  dt = std::max(config_.min_time_step, std::min(config_.max_time_step, dt));

  return dt;
}

// MultiLevelAdaptivityManager
MultiLevelAdaptivityManager::MultiLevelAdaptivityManager(const Config& config)
    : config_(config),
      hierarchy_(MeshHierarchy::Config{}) {

  error_estimator_ = std::make_unique<MultiGridErrorEstimator>();
  marker_ = std::make_unique<MultiLevelMarker>();
  fmg_solver_ = std::make_unique<FullMultiGridAdaptivity>();

  if (config_.use_octree) {
    octree_ = std::make_unique<OctreeAdaptivity>();
  }

  if (config_.use_spacetime) {
    spacetime_ = std::make_unique<SpaceTimeAdaptivity>();
  }
}

AdaptivityResult MultiLevelAdaptivityManager::adapt_multilevel(
    MeshHierarchy& hierarchy,
    const AdaptivityOptions& options) {

  AdaptivityResult result;

  // Estimate error on all levels
  std::vector<std::vector<double>> errors;
  for (size_t level = 0; level < hierarchy.num_levels(); ++level) {
    auto level_error = error_estimator_->estimate_error(hierarchy, level);
    errors.push_back(level_error);
  }

  // Mark elements on all levels
  auto marks = marker_->mark_hierarchy(hierarchy, errors);

  // Adapt each level
  for (size_t level = 0; level < hierarchy.num_levels(); ++level) {
    adapt_level(level);
  }

  // Synchronize levels
  synchronize_levels();

  // Update transfer operators
  update_operators();

  result.adaptation_complete = true;
  return result;
}

void MultiLevelAdaptivityManager::build_hierarchy(const MeshBase& initial_mesh) {
  hierarchy_.build_from_fine(initial_mesh);
}

void MultiLevelAdaptivityManager::adapt_level(size_t level) {
  // Adapt single level using standard adaptivity
  // Would use AdaptivityManager for each level
}

void MultiLevelAdaptivityManager::synchronize_levels() {
  // Ensure consistency between levels after adaptation
}

void MultiLevelAdaptivityManager::update_operators() {
  // Rebuild transfer operators after mesh changes
  for (size_t level = 1; level < hierarchy_.num_levels(); ++level) {
    hierarchy_.build_transfer_operators(level - 1, level);
  }
}

} // namespace svmp