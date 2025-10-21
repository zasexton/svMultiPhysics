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

#include "FieldTransfer.h"
#include "../MeshBase.h"
#include "../MeshFields.h"
#include "Marker.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace svmp {

namespace {

// Helper functions
double compute_element_volume(const MeshBase& mesh, size_t elem_id) {
  auto vertices = mesh.get_element_vertices(elem_id);

  if (vertices.size() == 3) {
    // Triangle area
    double x1 = vertices[0][0], y1 = vertices[0][1];
    double x2 = vertices[1][0], y2 = vertices[1][1];
    double x3 = vertices[2][0], y3 = vertices[2][1];
    return 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
  } else if (vertices.size() == 4) {
    // Quad area or tet volume
    double x1 = vertices[0][0], y1 = vertices[0][1], z1 = vertices[0][2];
    double x2 = vertices[1][0], y2 = vertices[1][1], z2 = vertices[1][2];
    double x3 = vertices[2][0], y3 = vertices[2][1], z3 = vertices[2][2];
    double x4 = vertices[3][0], y4 = vertices[3][1], z4 = vertices[3][2];

    // Check if coplanar (quad) or 3D (tet)
    double v1x = x2 - x1, v1y = y2 - y1, v1z = z2 - z1;
    double v2x = x3 - x1, v2y = y3 - y1, v2z = z3 - z1;
    double v3x = x4 - x1, v3y = y4 - y1, v3z = z4 - z1;

    // Cross product v2 x v3
    double cx = v2y * v3z - v2z * v3y;
    double cy = v2z * v3x - v2x * v3z;
    double cz = v2x * v3y - v2y * v3x;

    // Dot product v1 . (v2 x v3)
    double volume = std::abs(v1x * cx + v1y * cy + v1z * cz) / 6.0;

    if (volume < 1e-12) {
      // Coplanar - compute quad area
      double area1 = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
      double area2 = 0.5 * std::abs((x3 - x1) * (y4 - y1) - (x4 - x1) * (y3 - y1));
      return area1 + area2;
    }

    return volume;
  } else if (vertices.size() == 8) {
    // Hex volume - approximate as sum of 6 tets
    // Simplified calculation
    double dx = vertices[1][0] - vertices[0][0];
    double dy = vertices[2][1] - vertices[0][1];
    double dz = vertices[4][2] - vertices[0][2];
    return std::abs(dx * dy * dz);
  }

  return 1.0; // Default
}

double interpolate_at_point(const std::vector<double>& field,
                             const std::vector<std::pair<size_t, double>>& weights) {
  double value = 0.0;
  for (const auto& [idx, weight] : weights) {
    value += field[idx] * weight;
  }
  return value;
}

} // anonymous namespace

//=============================================================================
// LinearInterpolationTransfer Implementation
//=============================================================================

LinearInterpolationTransfer::LinearInterpolationTransfer(const Config& config)
    : config_(config) {}

TransferStats LinearInterpolationTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {

  auto start_time = std::chrono::steady_clock::now();
  TransferStats stats;

  // Transfer each field
  for (const auto& [name, old_field] : old_fields.get_fields()) {
    // Create new field
    new_fields.add_field(name, old_field.type, old_field.num_components);
    auto& new_field = new_fields.get_field(name);

    // Transfer based on field type
    if (old_field.type == FieldType::NODAL) {
      // Transfer nodal field
      if (parent_child.child_vertex_weights.empty()) {
        // Direct copy for unchanged vertices
        size_t num_nodes = std::min(old_mesh.num_nodes(), new_mesh.num_nodes());
        for (size_t i = 0; i < num_nodes; ++i) {
          for (size_t comp = 0; comp < old_field.num_components; ++comp) {
            new_field.values[i * old_field.num_components + comp] =
                old_field.values[i * old_field.num_components + comp];
          }
        }
      } else {
        // Interpolate at new vertices
        prolongate(old_mesh, new_mesh, old_field.values, new_field.values,
                   parent_child);
        stats.num_prolongations++;
      }
    } else if (old_field.type == FieldType::ELEMENTAL) {
      // Transfer elemental field
      if (parent_child.child_to_parent.empty()) {
        // Direct copy for unchanged elements
        size_t num_elems = std::min(old_mesh.num_elements(),
                                     new_mesh.num_elements());
        for (size_t i = 0; i < num_elems; ++i) {
          for (size_t comp = 0; comp < old_field.num_components; ++comp) {
            new_field.values[i * old_field.num_components + comp] =
                old_field.values[i * old_field.num_components + comp];
          }
        }
      } else if (parent_child.child_to_parent.size() > old_mesh.num_elements()) {
        // Refinement - prolongate
        prolongate(old_mesh, new_mesh, old_field.values, new_field.values,
                   parent_child);
        stats.num_prolongations++;
      } else {
        // Coarsening - restrict
        restrict(old_mesh, new_mesh, old_field.values, new_field.values,
                 parent_child);
        stats.num_restrictions++;
      }
    }

    // Check conservation if requested
    if (options.field_transfer == FieldTransferType::CONSERVATIVE) {
      double old_integral = compute_integral(old_mesh, old_field.values);
      double new_integral = compute_integral(new_mesh, new_field.values);
      stats.conservation_errors[name] = std::abs(new_integral - old_integral) /
                                         (std::abs(old_integral) + 1e-12);
    }

    stats.num_fields++;
  }

  auto end_time = std::chrono::steady_clock::now();
  stats.transfer_time = std::chrono::duration<double>(end_time - start_time).count();

  return stats;
}

void LinearInterpolationTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  // Initialize new field
  new_field.resize(new_mesh.num_nodes());

  // For vertices with interpolation weights
  for (const auto& [child_vertex, weights] : parent_child.child_vertex_weights) {
    if (child_vertex < new_field.size()) {
      new_field[child_vertex] = interpolate_at_vertex(old_field, weights);
    }
  }

  // For elements (if elemental field)
  if (old_field.size() == old_mesh.num_elements()) {
    new_field.resize(new_mesh.num_elements());

    for (size_t child_elem = 0; child_elem < parent_child.child_to_parent.size();
         ++child_elem) {
      size_t parent_elem = parent_child.child_to_parent[child_elem];
      if (parent_elem < old_field.size() && child_elem < new_field.size()) {
        // Direct injection from parent
        new_field[child_elem] = old_field[parent_elem];
      }
    }
  }

  // Handle boundary preservation
  if (config_.preserve_boundary) {
    // Copy boundary values directly
    for (size_t i = 0; i < old_mesh.num_nodes(); ++i) {
      if (old_mesh.is_boundary_node(i) && i < new_field.size()) {
        new_field[i] = old_field[i];
      }
    }
  }
}

void LinearInterpolationTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  // Initialize new field
  new_field.resize(new_mesh.num_nodes(), 0.0);

  // For parent vertices that have children
  for (const auto& [parent_vertex, children] :
       parent_child.parent_vertex_to_children) {
    if (parent_vertex < new_field.size()) {
      // Average values from children
      double sum = 0.0;
      size_t count = 0;

      for (size_t child : children) {
        if (child < old_field.size()) {
          sum += old_field[child];
          count++;
        }
      }

      if (count > 0) {
        new_field[parent_vertex] = sum / count;
      }
    }
  }

  // For elements (if elemental field)
  if (old_field.size() == old_mesh.num_elements()) {
    new_field.resize(new_mesh.num_elements(), 0.0);

    for (const auto& [parent_elem, children] : parent_child.parent_to_children) {
      if (parent_elem < new_field.size()) {
        new_field[parent_elem] = average_from_children(old_field, children,
                                                        old_mesh);
      }
    }
  }
}

double LinearInterpolationTransfer::interpolate_at_vertex(
    const std::vector<double>& old_field,
    const std::vector<std::pair<size_t, double>>& weights) const {

  double value = 0.0;
  double weight_sum = 0.0;

  for (const auto& [vertex_id, weight] : weights) {
    if (vertex_id < old_field.size() && weight > config_.min_weight) {
      value += old_field[vertex_id] * weight;
      weight_sum += weight;
    }
  }

  if (weight_sum > config_.min_weight) {
    return value / weight_sum;
  }

  return 0.0;
}

double LinearInterpolationTransfer::average_from_children(
    const std::vector<double>& old_field,
    const std::vector<size_t>& children,
    const MeshBase& mesh) const {

  if (config_.use_volume_weighting) {
    // Volume-weighted average
    double weighted_sum = 0.0;
    double volume_sum = 0.0;

    for (size_t child : children) {
      if (child < old_field.size()) {
        double volume = compute_element_volume(mesh, child);
        weighted_sum += old_field[child] * volume;
        volume_sum += volume;
      }
    }

    if (volume_sum > 1e-12) {
      return weighted_sum / volume_sum;
    }
  }

  // Simple average
  double sum = 0.0;
  size_t count = 0;

  for (size_t child : children) {
    if (child < old_field.size()) {
      sum += old_field[child];
      count++;
    }
  }

  return count > 0 ? sum / count : 0.0;
}

//=============================================================================
// ConservativeTransfer Implementation
//=============================================================================

ConservativeTransfer::ConservativeTransfer(const Config& config)
    : config_(config) {}

TransferStats ConservativeTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {

  auto start_time = std::chrono::steady_clock::now();
  TransferStats stats;

  // Transfer each field
  for (const auto& [name, old_field] : old_fields.get_fields()) {
    // Create new field
    new_fields.add_field(name, old_field.type, old_field.num_components);
    auto& new_field = new_fields.get_field(name);

    // Compute old integral
    double old_integral = compute_integral(old_mesh, old_field.values);

    // Transfer field
    if (parent_child.child_to_parent.size() > old_mesh.num_elements()) {
      // Refinement
      prolongate(old_mesh, new_mesh, old_field.values, new_field.values,
                 parent_child);
      stats.num_prolongations++;
    } else {
      // Coarsening
      restrict(old_mesh, new_mesh, old_field.values, new_field.values,
                parent_child);
      stats.num_restrictions++;
    }

    // Enforce conservation
    enforce_conservation(old_mesh, new_mesh, old_field.values, new_field.values);

    // Compute conservation error
    double new_integral = compute_integral(new_mesh, new_field.values);
    stats.conservation_errors[name] = std::abs(new_integral - old_integral) /
                                       (std::abs(old_integral) + 1e-12);

    stats.num_fields++;
  }

  auto end_time = std::chrono::steady_clock::now();
  stats.transfer_time = std::chrono::duration<double>(end_time - start_time).count();

  return stats;
}

void ConservativeTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  new_field.resize(new_mesh.num_elements(), 0.0);

  // For each parent element
  for (const auto& [parent_elem, children] : parent_child.parent_to_children) {
    if (parent_elem >= old_field.size()) continue;

    // Get parent value and volume
    double parent_value = old_field[parent_elem];
    double parent_volume = compute_element_volume(old_mesh, parent_elem);

    if (config_.high_order_reconstruction) {
      // High-order reconstruction in parent
      auto reconstructed = reconstruct_in_parent(old_mesh, parent_elem,
                                                  old_field);

      // Evaluate at child centroids
      for (size_t child : children) {
        if (child < new_field.size()) {
          // For now, use parent value (would evaluate polynomial)
          new_field[child] = parent_value;
        }
      }
    } else {
      // Distribute conservatively to children
      double total_child_volume = 0.0;
      for (size_t child : children) {
        total_child_volume += compute_element_volume(new_mesh, child);
      }

      // Distribute based on volume fraction
      for (size_t child : children) {
        if (child < new_field.size()) {
          double child_volume = compute_element_volume(new_mesh, child);
          new_field[child] = parent_value * parent_volume * child_volume /
                             (total_child_volume * child_volume);
        }
      }
    }
  }
}

void ConservativeTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  new_field.resize(new_mesh.num_elements(), 0.0);

  // For each parent element
  for (const auto& [parent_elem, children] : parent_child.parent_to_children) {
    if (parent_elem >= new_field.size()) continue;

    // Compute integral over children
    double child_integral = 0.0;
    double child_volume = 0.0;

    for (size_t child : children) {
      if (child < old_field.size()) {
        double vol = compute_element_volume(old_mesh, child);
        child_integral += old_field[child] * vol;
        child_volume += vol;
      }
    }

    // Set parent value to preserve integral
    double parent_volume = compute_element_volume(new_mesh, parent_elem);
    if (parent_volume > 1e-12) {
      new_field[parent_elem] = child_integral / parent_volume;
    }
  }
}

double ConservativeTransfer::compute_integral(
    const MeshBase& mesh,
    const std::vector<double>& field) const {

  double integral = 0.0;

  if (field.size() == mesh.num_nodes()) {
    // Nodal field - integrate over elements
    size_t num_elements = mesh.num_elements();
    for (size_t elem = 0; elem < num_elements; ++elem) {
      auto vertices = mesh.get_element_vertex_ids(elem);
      double elem_volume = compute_element_volume(mesh, elem);

      // Average nodal values
      double avg_value = 0.0;
      for (size_t v : vertices) {
        if (v < field.size()) {
          avg_value += field[v];
        }
      }
      avg_value /= vertices.size();

      integral += avg_value * elem_volume;
    }
  } else if (field.size() == mesh.num_elements()) {
    // Elemental field
    for (size_t elem = 0; elem < field.size(); ++elem) {
      double elem_volume = compute_element_volume(mesh, elem);
      integral += field[elem] * elem_volume;
    }
  }

  return integral;
}

void ConservativeTransfer::enforce_conservation(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field) const {

  // Compute integrals
  double old_integral = compute_integral(old_mesh, old_field);
  double new_integral = compute_integral(new_mesh, new_field);

  if (std::abs(new_integral) < 1e-12) return;

  // Check if conservation is satisfied
  double error = std::abs(new_integral - old_integral) / std::abs(old_integral + 1e-12);

  if (error > config_.conservation_tolerance) {
    // Apply correction factor
    double correction = old_integral / new_integral;

    // Iterative correction
    for (size_t iter = 0; iter < config_.max_conservation_iterations; ++iter) {
      // Scale field
      for (double& value : new_field) {
        value *= correction;
      }

      // Check convergence
      new_integral = compute_integral(new_mesh, new_field);
      error = std::abs(new_integral - old_integral) / std::abs(old_integral + 1e-12);

      if (error < config_.conservation_tolerance) {
        break;
      }

      // Update correction
      correction = old_integral / new_integral;
    }
  }
}

std::vector<double> ConservativeTransfer::reconstruct_in_parent(
    const MeshBase& mesh,
    size_t parent_elem,
    const std::vector<double>& field) const {

  // Simplified reconstruction
  // Would implement polynomial reconstruction for high-order accuracy
  std::vector<double> coefficients;

  // For now, return constant (0-th order)
  if (parent_elem < field.size()) {
    coefficients.push_back(field[parent_elem]);
  }

  return coefficients;
}

//=============================================================================
// HighOrderTransfer Implementation
//=============================================================================

HighOrderTransfer::HighOrderTransfer(const Config& config)
    : config_(config) {}

TransferStats HighOrderTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {

  auto start_time = std::chrono::steady_clock::now();
  TransferStats stats;

  // Transfer each field
  for (const auto& [name, old_field] : old_fields.get_fields()) {
    // Create new field
    new_fields.add_field(name, old_field.type, old_field.num_components);
    auto& new_field = new_fields.get_field(name);

    // Transfer using high-order reconstruction
    if (parent_child.child_to_parent.size() > old_mesh.num_elements()) {
      // Refinement
      prolongate(old_mesh, new_mesh, old_field.values, new_field.values,
                 parent_child);
      stats.num_prolongations++;
    } else {
      // Coarsening
      restrict(old_mesh, new_mesh, old_field.values, new_field.values,
                parent_child);
      stats.num_restrictions++;
    }

    stats.num_fields++;
  }

  auto end_time = std::chrono::steady_clock::now();
  stats.transfer_time = std::chrono::duration<double>(end_time - start_time).count();

  return stats;
}

void HighOrderTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  new_field.resize(new_mesh.num_elements(), 0.0);

  // For each parent element
  for (const auto& [parent_elem, children] : parent_child.parent_to_children) {
    if (parent_elem >= old_field.size()) continue;

    // Build polynomial reconstruction
    auto polynomial = build_polynomial(old_mesh, parent_elem, old_field);

    // Evaluate at child element centers
    for (size_t child : children) {
      if (child < new_field.size()) {
        auto child_vertices = new_mesh.get_element_vertices(child);

        // Compute child centroid
        std::array<double, 3> centroid = {0, 0, 0};
        for (const auto& v : child_vertices) {
          for (int i = 0; i < 3; ++i) {
            centroid[i] += v[i];
          }
        }
        for (int i = 0; i < 3; ++i) {
          centroid[i] /= child_vertices.size();
        }

        // Evaluate polynomial at centroid
        new_field[child] = evaluate_polynomial(polynomial, centroid);

        // Apply gradient limiting if needed
        if (config_.limit_gradients && polynomial.size() > 1) {
          // Check for oscillations and limit if necessary
          auto parent_vertices = old_mesh.get_element_vertices(parent_elem);
          double parent_value = old_field[parent_elem];

          // Find min/max in stencil
          double min_val = parent_value;
          double max_val = parent_value;

          // Get neighbor elements for stencil
          auto neighbors = old_mesh.get_element_neighbors(parent_elem);
          for (size_t neighbor : neighbors) {
            if (neighbor < old_field.size()) {
              min_val = std::min(min_val, old_field[neighbor]);
              max_val = std::max(max_val, old_field[neighbor]);
            }
          }

          // Apply limiter
          new_field[child] = std::max(min_val, std::min(max_val, new_field[child]));
        }
      }
    }
  }
}

void HighOrderTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  new_field.resize(new_mesh.num_elements(), 0.0);

  // For each parent element
  for (const auto& [parent_elem, children] : parent_child.parent_to_children) {
    if (parent_elem >= new_field.size()) continue;

    if (config_.use_least_squares) {
      // Least squares fit to children
      std::vector<std::array<double, 3>> child_points;
      std::vector<double> child_values;

      for (size_t child : children) {
        if (child < old_field.size()) {
          auto vertices = old_mesh.get_element_vertices(child);

          // Use child centroid
          std::array<double, 3> centroid = {0, 0, 0};
          for (const auto& v : vertices) {
            for (int i = 0; i < 3; ++i) {
              centroid[i] += v[i];
            }
          }
          for (int i = 0; i < 3; ++i) {
            centroid[i] /= vertices.size();
          }

          child_points.push_back(centroid);
          child_values.push_back(old_field[child]);
        }
      }

      // Fit polynomial and evaluate at parent center
      if (!child_points.empty()) {
        // For simplicity, use average (0-th order fit)
        double sum = 0.0;
        for (double val : child_values) {
          sum += val;
        }
        new_field[parent_elem] = sum / child_values.size();
      }
    } else {
      // Simple averaging
      double sum = 0.0;
      size_t count = 0;

      for (size_t child : children) {
        if (child < old_field.size()) {
          sum += old_field[child];
          count++;
        }
      }

      if (count > 0) {
        new_field[parent_elem] = sum / count;
      }
    }
  }
}

std::vector<double> HighOrderTransfer::build_polynomial(
    const MeshBase& mesh,
    size_t elem_id,
    const std::vector<double>& field) const {

  std::vector<double> coefficients;

  // Get stencil elements
  std::vector<size_t> stencil;
  stencil.push_back(elem_id);

  // Add neighbors to stencil
  auto neighbors = mesh.get_element_neighbors(elem_id);
  for (size_t neighbor : neighbors) {
    stencil.push_back(neighbor);
    if (stencil.size() >= config_.min_stencil_size) {
      break;
    }
  }

  // Add neighbors of neighbors if needed
  if (stencil.size() < config_.min_stencil_size) {
    std::vector<size_t> second_neighbors;
    for (size_t n1 : neighbors) {
      auto n2_list = mesh.get_element_neighbors(n1);
      for (size_t n2 : n2_list) {
        if (std::find(stencil.begin(), stencil.end(), n2) == stencil.end()) {
          second_neighbors.push_back(n2);
          if (stencil.size() + second_neighbors.size() >= config_.min_stencil_size) {
            break;
          }
        }
      }
    }
    stencil.insert(stencil.end(), second_neighbors.begin(), second_neighbors.end());
  }

  // Build polynomial coefficients
  // For now, return simple average (0-th order)
  double sum = 0.0;
  size_t count = 0;

  for (size_t stencil_elem : stencil) {
    if (stencil_elem < field.size()) {
      sum += field[stencil_elem];
      count++;
    }
  }

  coefficients.push_back(count > 0 ? sum / count : 0.0);

  // For higher order, would compute gradients, etc.
  if (config_.polynomial_order >= 1) {
    // Add gradient terms (simplified)
    coefficients.push_back(0.0); // dx
    coefficients.push_back(0.0); // dy
    coefficients.push_back(0.0); // dz
  }

  if (config_.polynomial_order >= 2) {
    // Add second-order terms
    coefficients.push_back(0.0); // dxx
    coefficients.push_back(0.0); // dyy
    coefficients.push_back(0.0); // dzz
    coefficients.push_back(0.0); // dxy
    coefficients.push_back(0.0); // dxz
    coefficients.push_back(0.0); // dyz
  }

  return coefficients;
}

double HighOrderTransfer::evaluate_polynomial(
    const std::vector<double>& coefficients,
    const std::array<double, 3>& point) const {

  if (coefficients.empty()) return 0.0;

  double value = coefficients[0]; // Constant term

  if (coefficients.size() > 1 && config_.polynomial_order >= 1) {
    // Linear terms
    value += coefficients[1] * point[0];
    value += coefficients[2] * point[1];
    value += coefficients[3] * point[2];
  }

  if (coefficients.size() > 4 && config_.polynomial_order >= 2) {
    // Quadratic terms
    value += coefficients[4] * point[0] * point[0];
    value += coefficients[5] * point[1] * point[1];
    value += coefficients[6] * point[2] * point[2];
    value += coefficients[7] * point[0] * point[1];
    value += coefficients[8] * point[0] * point[2];
    value += coefficients[9] * point[1] * point[2];
  }

  return value;
}

void HighOrderTransfer::apply_limiter(
    std::vector<double>& gradients,
    const MeshBase& mesh,
    size_t elem_id) const {

  // Simple gradient limiter
  // Would implement Barth-Jespersen or Venkatakrishnan limiter

  // Get element value range from neighbors
  auto neighbors = mesh.get_element_neighbors(elem_id);

  // For now, just scale gradients if they're too large
  double max_gradient = 1.0;
  for (double& grad : gradients) {
    if (std::abs(grad) > max_gradient) {
      grad = grad > 0 ? max_gradient : -max_gradient;
    }
  }
}

//=============================================================================
// InjectionTransfer Implementation
//=============================================================================

TransferStats InjectionTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {

  auto start_time = std::chrono::steady_clock::now();
  TransferStats stats;

  // Transfer each field
  for (const auto& [name, old_field] : old_fields.get_fields()) {
    // Create new field
    new_fields.add_field(name, old_field.type, old_field.num_components);
    auto& new_field = new_fields.get_field(name);

    // Direct injection
    if (parent_child.child_to_parent.size() > old_mesh.num_elements()) {
      // Refinement
      prolongate(old_mesh, new_mesh, old_field.values, new_field.values,
                 parent_child);
      stats.num_prolongations++;
    } else {
      // Coarsening
      restrict(old_mesh, new_mesh, old_field.values, new_field.values,
                parent_child);
      stats.num_restrictions++;
    }

    stats.num_fields++;
  }

  auto end_time = std::chrono::steady_clock::now();
  stats.transfer_time = std::chrono::duration<double>(end_time - start_time).count();

  return stats;
}

void InjectionTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  new_field.resize(new_mesh.num_elements(), 0.0);

  // Direct injection from parent to children
  for (size_t child = 0; child < parent_child.child_to_parent.size(); ++child) {
    size_t parent = parent_child.child_to_parent[child];
    if (parent < old_field.size() && child < new_field.size()) {
      new_field[child] = old_field[parent];
    }
  }
}

void InjectionTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {

  new_field.resize(new_mesh.num_elements(), 0.0);

  // Direct injection from first child to parent
  for (const auto& [parent, children] : parent_child.parent_to_children) {
    if (!children.empty() && parent < new_field.size()) {
      size_t first_child = children[0];
      if (first_child < old_field.size()) {
        new_field[parent] = old_field[first_child];
      }
    }
  }
}

//=============================================================================
// FieldTransferFactory Implementation
//=============================================================================

std::unique_ptr<FieldTransfer> FieldTransferFactory::create(
    const AdaptivityOptions& options) {

  switch (options.field_transfer) {
    case FieldTransferType::LINEAR_INTERPOLATION:
      return create_linear();

    case FieldTransferType::CONSERVATIVE:
      return create_conservative();

    case FieldTransferType::HIGH_ORDER:
      return create_high_order();

    case FieldTransferType::INJECTION:
      return create_injection();

    default:
      return create_linear();
  }
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_linear(
    const LinearInterpolationTransfer::Config& config) {
  return std::make_unique<LinearInterpolationTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_conservative(
    const ConservativeTransfer::Config& config) {
  return std::make_unique<ConservativeTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_high_order(
    const HighOrderTransfer::Config& config) {
  return std::make_unique<HighOrderTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_injection() {
  return std::make_unique<InjectionTransfer>();
}

//=============================================================================
// FieldTransferUtils Implementation
//=============================================================================

ParentChildMap FieldTransferUtils::build_parent_child_map(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<MarkType>& marks) {

  ParentChildMap map;

  // Build element parent-child relationships
  for (size_t elem = 0; elem < marks.size(); ++elem) {
    if (marks[elem] == MarkType::REFINE) {
      // This element was refined
      // Find its children in the new mesh
      // (Simplified - would need actual topology tracking)
      std::vector<size_t> children;

      // Assume regular refinement creates 4 children for 2D, 8 for 3D
      size_t num_children = (old_mesh.get_element_type(elem) == ElementType::TETRAHEDRON ||
                              old_mesh.get_element_type(elem) == ElementType::HEXAHEDRON) ? 8 : 4;

      size_t child_start = elem * num_children; // Simplified indexing
      for (size_t i = 0; i < num_children; ++i) {
        size_t child = child_start + i;
        if (child < new_mesh.num_elements()) {
          children.push_back(child);
          map.child_to_parent[child] = elem;
        }
      }

      map.parent_to_children[elem] = children;
    }
  }

  // Build vertex interpolation weights (simplified)
  // Would need actual geometric interpolation
  size_t num_old_vertices = old_mesh.num_nodes();
  size_t num_new_vertices = new_mesh.num_nodes();

  if (num_new_vertices > num_old_vertices) {
    // New vertices created by refinement
    for (size_t v = num_old_vertices; v < num_new_vertices; ++v) {
      // Find parent vertices for interpolation
      // Simplified: use two nearest old vertices
      std::vector<std::pair<size_t, double>> weights;

      if (v % 2 == 0 && v > 1) {
        // Edge midpoint
        size_t v1 = (v - num_old_vertices) / 2;
        size_t v2 = v1 + 1;
        if (v1 < num_old_vertices && v2 < num_old_vertices) {
          weights.push_back({v1, 0.5});
          weights.push_back({v2, 0.5});
        }
      }

      if (!weights.empty()) {
        map.child_vertex_weights[v] = weights;
      }
    }
  }

  return map;
}

double FieldTransferUtils::check_conservation(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    const std::vector<double>& new_field) {

  // Compute integrals
  double old_integral = 0.0;
  double new_integral = 0.0;

  if (old_field.size() == old_mesh.num_elements()) {
    // Elemental field
    for (size_t elem = 0; elem < old_field.size(); ++elem) {
      double volume = compute_element_volume(old_mesh, elem);
      old_integral += old_field[elem] * volume;
    }

    for (size_t elem = 0; elem < new_field.size(); ++elem) {
      double volume = compute_element_volume(new_mesh, elem);
      new_integral += new_field[elem] * volume;
    }
  } else {
    // Nodal field - integrate over elements
    size_t num_old_elements = old_mesh.num_elements();
    for (size_t elem = 0; elem < num_old_elements; ++elem) {
      auto vertices = old_mesh.get_element_vertex_ids(elem);
      double volume = compute_element_volume(old_mesh, elem);

      double avg_value = 0.0;
      for (size_t v : vertices) {
        if (v < old_field.size()) {
          avg_value += old_field[v];
        }
      }
      avg_value /= vertices.size();

      old_integral += avg_value * volume;
    }

    size_t num_new_elements = new_mesh.num_elements();
    for (size_t elem = 0; elem < num_new_elements; ++elem) {
      auto vertices = new_mesh.get_element_vertex_ids(elem);
      double volume = compute_element_volume(new_mesh, elem);

      double avg_value = 0.0;
      for (size_t v : vertices) {
        if (v < new_field.size()) {
          avg_value += new_field[v];
        }
      }
      avg_value /= vertices.size();

      new_integral += avg_value * volume;
    }
  }

  // Return relative error
  if (std::abs(old_integral) < 1e-12) return 0.0;

  return std::abs(new_integral - old_integral) / std::abs(old_integral);
}

double FieldTransferUtils::compute_interpolation_error(
    const MeshBase& mesh,
    const std::vector<double>& exact_field,
    const std::vector<double>& interpolated_field) {

  if (exact_field.size() != interpolated_field.size()) {
    return -1.0; // Invalid comparison
  }

  double max_error = 0.0;
  double sum_sq_error = 0.0;
  double sum_sq_exact = 0.0;

  for (size_t i = 0; i < exact_field.size(); ++i) {
    double error = std::abs(interpolated_field[i] - exact_field[i]);
    max_error = std::max(max_error, error);
    sum_sq_error += error * error;
    sum_sq_exact += exact_field[i] * exact_field[i];
  }

  // Return relative L2 error
  if (sum_sq_exact < 1e-12) return max_error;

  return std::sqrt(sum_sq_error / sum_sq_exact);
}

void FieldTransferUtils::project_field(
    const MeshBase& source_mesh,
    const MeshBase& target_mesh,
    const std::vector<double>& source_field,
    std::vector<double>& target_field) {

  // General field projection between non-nested meshes
  // Would implement L2 projection or interpolation search

  target_field.resize(target_mesh.num_nodes(), 0.0);

  // For each target node, find containing source element
  for (size_t target_node = 0; target_node < target_mesh.num_nodes(); ++target_node) {
    auto target_pos = target_mesh.get_node_position(target_node);

    // Find source element containing this point (simplified)
    // Would use spatial search structure
    bool found = false;

    for (size_t source_elem = 0; source_elem < source_mesh.num_elements(); ++source_elem) {
      auto vertices = source_mesh.get_element_vertices(source_elem);

      // Check if point is in element (simplified)
      // Would use proper containment test

      if (found) {
        // Interpolate within element
        if (source_field.size() == source_mesh.num_elements()) {
          // Elemental field
          target_field[target_node] = source_field[source_elem];
        } else {
          // Nodal field - interpolate from element vertices
          auto vertex_ids = source_mesh.get_element_vertex_ids(source_elem);
          double sum = 0.0;
          for (size_t v : vertex_ids) {
            if (v < source_field.size()) {
              sum += source_field[v];
            }
          }
          target_field[target_node] = sum / vertex_ids.size();
        }
        break;
      }
    }

    // If not found, use nearest neighbor (fallback)
    if (!found && target_node < source_field.size()) {
      target_field[target_node] = source_field[target_node];
    }
  }
}

TransferStats FieldTransferUtils::transfer_all_fields(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) {

  // Create appropriate transfer strategy
  auto transfer = FieldTransferFactory::create(options);

  // Transfer all fields
  return transfer->transfer(old_mesh, new_mesh, old_fields, new_fields,
                            parent_child, options);
}

} // namespace svmp