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

#include "AnisotropicAdaptivity.h"
#include "../MeshBase.h"
#include "../MeshFields.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <Eigen/Dense>

namespace svmp {

//=============================================================================
// MetricTensor Implementation
//=============================================================================

void MetricTensor::compute_eigendecomposition() {
  if (is_3d) {
    // 3D metric eigendecomposition
    Eigen::Matrix3d M;
    M << metric_3d[0], metric_3d[1], metric_3d[2],
         metric_3d[1], metric_3d[3], metric_3d[4],
         metric_3d[2], metric_3d[4], metric_3d[5];

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
    auto evalues = solver.eigenvalues();
    auto evectors = solver.eigenvectors();

    for (int i = 0; i < 3; ++i) {
      eigenvalues[i] = evalues[i];
      for (int j = 0; j < 3; ++j) {
        eigenvectors[i][j] = evectors(j, i);
      }
    }
  } else {
    // 2D metric eigendecomposition
    Eigen::Matrix2d M;
    M << metric_2d[0], metric_2d[1],
         metric_2d[1], metric_2d[2];

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M);
    auto evalues = solver.eigenvalues();
    auto evectors = solver.eigenvectors();

    eigenvalues[0] = evalues[0];
    eigenvalues[1] = evalues[1];
    eigenvalues[2] = 1.0;

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        eigenvectors[i][j] = evectors(j, i);
      }
    }
    eigenvectors[2] = {0, 0, 1};
  }

  // Compute anisotropy ratio
  double min_eval = *std::min_element(eigenvalues.begin(), eigenvalues.end());
  double max_eval = *std::max_element(eigenvalues.begin(), eigenvalues.end());
  anisotropy_ratio = (min_eval > 0) ? max_eval / min_eval : 1.0;
}

double MetricTensor::evaluate_in_direction(const std::array<double, 3>& direction) const {
  if (is_3d) {
    // M(v) = v^T * M * v
    double result = metric_3d[0] * direction[0] * direction[0] +
                    metric_3d[3] * direction[1] * direction[1] +
                    metric_3d[5] * direction[2] * direction[2] +
                    2 * metric_3d[1] * direction[0] * direction[1] +
                    2 * metric_3d[2] * direction[0] * direction[2] +
                    2 * metric_3d[4] * direction[1] * direction[2];
    return result;
  } else {
    double result = metric_2d[0] * direction[0] * direction[0] +
                    metric_2d[2] * direction[1] * direction[1] +
                    2 * metric_2d[1] * direction[0] * direction[1];
    return result;
  }
}

MetricTensor MetricTensor::interpolate(const MetricTensor& m1, const MetricTensor& m2, double t) {
  MetricTensor result;
  result.is_3d = m1.is_3d;

  if (result.is_3d) {
    for (int i = 0; i < 6; ++i) {
      result.metric_3d[i] = (1 - t) * m1.metric_3d[i] + t * m2.metric_3d[i];
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      result.metric_2d[i] = (1 - t) * m1.metric_2d[i] + t * m2.metric_2d[i];
    }
  }

  result.compute_eigendecomposition();
  return result;
}

MetricTensor MetricTensor::intersect(const MetricTensor& m1, const MetricTensor& m2) {
  // Metric intersection: take maximum eigenvalues (minimum spacing)
  MetricTensor result;
  result.is_3d = m1.is_3d;

  // Use eigendecomposition approach
  MetricTensor m1_copy = m1, m2_copy = m2;
  m1_copy.compute_eigendecomposition();
  m2_copy.compute_eigendecomposition();

  // Take maximum eigenvalues
  for (int i = 0; i < 3; ++i) {
    result.eigenvalues[i] = std::max(m1_copy.eigenvalues[i], m2_copy.eigenvalues[i]);
  }

  // Use average eigenvectors (simplified)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result.eigenvectors[i][j] = 0.5 * (m1_copy.eigenvectors[i][j] +
                                          m2_copy.eigenvectors[i][j]);
    }
  }

  // Reconstruct metric from eigendecomposition
  if (result.is_3d) {
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d evec(result.eigenvectors[i][0],
                           result.eigenvectors[i][1],
                           result.eigenvectors[i][2]);
      M += result.eigenvalues[i] * evec * evec.transpose();
    }
    result.metric_3d[0] = M(0, 0);
    result.metric_3d[1] = M(0, 1);
    result.metric_3d[2] = M(0, 2);
    result.metric_3d[3] = M(1, 1);
    result.metric_3d[4] = M(1, 2);
    result.metric_3d[5] = M(2, 2);
  }

  return result;
}

//=============================================================================
// AnisotropicErrorEstimator Implementation
//=============================================================================

AnisotropicErrorEstimator::AnisotropicErrorEstimator(const Config& config)
    : config_(config) {}

void AnisotropicErrorEstimator::estimate_error(
    const MeshBase& mesh,
    const MeshFields* fields,
    std::vector<double>& error_estimate) const {

  size_t num_elements = mesh.num_elements();
  error_estimate.resize(num_elements);

  // Compute metric field
  auto metric_field = compute_metric_field(mesh, fields);

  // Convert metric to error estimate
  for (size_t elem = 0; elem < num_elements; ++elem) {
    // Use maximum eigenvalue as error indicator
    metric_field[elem].compute_eigendecomposition();
    error_estimate[elem] = *std::max_element(
        metric_field[elem].eigenvalues.begin(),
        metric_field[elem].eigenvalues.end());
  }
}

std::vector<MetricTensor> AnisotropicErrorEstimator::compute_metric_field(
    const MeshBase& mesh,
    const MeshFields* fields) const {

  size_t num_elements = mesh.num_elements();
  std::vector<MetricTensor> metrics(num_elements);

  // Compute metric based on method
  for (size_t elem = 0; elem < num_elements; ++elem) {
    switch (config_.method) {
      case Config::Method::HESSIAN_BASED:
        metrics[elem] = compute_hessian_metric(mesh, elem, fields);
        break;
      case Config::Method::GRADIENT_BASED:
        metrics[elem] = compute_gradient_metric(mesh, elem, fields);
        break;
      default:
        metrics[elem] = compute_hessian_metric(mesh, elem, fields);
    }
  }

  // Apply gradation control
  if (config_.gradation_parameter > 1.0) {
    apply_gradation(mesh, metrics);
  }

  // Smooth metric field
  if (config_.smooth_metric) {
    smooth_metric_field(mesh, metrics);
  }

  // Normalize to target complexity
  normalize_metric_field(mesh, metrics);

  return metrics;
}

std::vector<std::array<double, 3>> AnisotropicErrorEstimator::compute_directional_errors(
    const MeshBase& mesh,
    const MeshFields* fields) const {

  size_t num_elements = mesh.num_elements();
  std::vector<std::array<double, 3>> directional_errors(num_elements);

  auto metric_field = compute_metric_field(mesh, fields);

  for (size_t elem = 0; elem < num_elements; ++elem) {
    metric_field[elem].compute_eigendecomposition();

    // Use eigenvalues as directional errors
    for (int i = 0; i < 3; ++i) {
      directional_errors[elem][i] = metric_field[elem].eigenvalues[i];
    }
  }

  return directional_errors;
}

MetricTensor AnisotropicErrorEstimator::compute_hessian_metric(
    const MeshBase& mesh,
    size_t element_id,
    const MeshFields* fields) const {

  MetricTensor metric;
  metric.is_3d = (mesh.get_dimension() == 3);

  if (!fields || fields->get_fields().empty()) {
    return metric; // Identity metric
  }

  // Get first field for Hessian computation
  const auto& field_data = fields->get_fields().begin()->second.values;

  // Compute Hessian
  auto hessian = compute_hessian(mesh, element_id, field_data);

  // Convert Hessian to metric
  if (metric.is_3d) {
    for (int i = 0; i < 6; ++i) {
      metric.metric_3d[i] = std::abs(hessian[i]);
    }
  } else {
    metric.metric_2d[0] = std::abs(hessian[0]);
    metric.metric_2d[1] = std::abs(hessian[1]);
    metric.metric_2d[2] = std::abs(hessian[3]);
  }

  metric.compute_eigendecomposition();

  // Apply bounds
  for (int i = 0; i < 3; ++i) {
    double h = 1.0 / std::sqrt(metric.eigenvalues[i]);
    h = std::max(config_.min_size, std::min(config_.max_size, h));
    metric.eigenvalues[i] = 1.0 / (h * h);
  }

  return metric;
}

MetricTensor AnisotropicErrorEstimator::compute_gradient_metric(
    const MeshBase& mesh,
    size_t element_id,
    const MeshFields* fields) const {

  MetricTensor metric;
  metric.is_3d = (mesh.get_dimension() == 3);

  if (!fields || fields->get_fields().empty()) {
    return metric;
  }

  // Compute gradient
  const auto& field_data = fields->get_fields().begin()->second.values;
  auto vertices = mesh.get_element_vertex_ids(element_id);

  std::array<double, 3> gradient = {0, 0, 0};

  // Simple gradient computation
  if (vertices.size() >= 3) {
    auto positions = mesh.get_element_vertices(element_id);

    // Compute gradient using least squares
    for (size_t i = 0; i < vertices.size(); ++i) {
      for (size_t j = i + 1; j < vertices.size(); ++j) {
        double df = field_data[vertices[j]] - field_data[vertices[i]];
        std::array<double, 3> dx = {
          positions[j][0] - positions[i][0],
          positions[j][1] - positions[i][1],
          positions[j][2] - positions[i][2]
        };
        double dist_sq = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
        if (dist_sq > 1e-12) {
          for (int k = 0; k < 3; ++k) {
            gradient[k] += df * dx[k] / dist_sq;
          }
        }
      }
    }
  }

  // Build metric aligned with gradient
  double grad_mag = std::sqrt(gradient[0]*gradient[0] +
                              gradient[1]*gradient[1] +
                              gradient[2]*gradient[2]);

  if (grad_mag > 1e-12) {
    // Normalize gradient direction
    for (int i = 0; i < 3; ++i) {
      gradient[i] /= grad_mag;
    }

    // Set strong refinement along gradient, weaker perpendicular
    metric.eigenvalues[0] = grad_mag * grad_mag;
    metric.eigenvalues[1] = grad_mag * 0.1;
    metric.eigenvalues[2] = grad_mag * 0.1;

    metric.eigenvectors[0] = gradient;

    // Compute perpendicular directions
    if (std::abs(gradient[0]) < 0.9) {
      metric.eigenvectors[1] = {1, 0, 0};
    } else {
      metric.eigenvectors[1] = {0, 1, 0};
    }

    // Cross product for third direction
    metric.eigenvectors[2] = {
      gradient[1] * metric.eigenvectors[1][2] - gradient[2] * metric.eigenvectors[1][1],
      gradient[2] * metric.eigenvectors[1][0] - gradient[0] * metric.eigenvectors[1][2],
      gradient[0] * metric.eigenvectors[1][1] - gradient[1] * metric.eigenvectors[1][0]
    };
  }

  return metric;
}

std::array<double, 6> AnisotropicErrorEstimator::compute_hessian(
    const MeshBase& mesh,
    size_t element_id,
    const std::vector<double>& field) const {

  // Simplified Hessian computation
  std::array<double, 6> hessian = {0};

  auto vertices = mesh.get_element_vertex_ids(element_id);
  auto positions = mesh.get_element_vertices(element_id);

  if (vertices.size() < 4) {
    // Need at least 4 points for Hessian
    return hessian;
  }

  // Use finite differences (simplified)
  double h = 1e-3; // Step size

  for (size_t i = 0; i < vertices.size(); ++i) {
    double f_i = field[vertices[i]];

    for (size_t j = i + 1; j < vertices.size(); ++j) {
      double f_j = field[vertices[j]];

      // Approximate second derivatives
      double dx = positions[j][0] - positions[i][0];
      double dy = positions[j][1] - positions[i][1];
      double dz = positions[j][2] - positions[i][2];

      if (std::abs(dx) > h) {
        hessian[0] += (f_j - f_i) / (dx * dx); // d²f/dx²
      }
      if (std::abs(dy) > h) {
        hessian[3] += (f_j - f_i) / (dy * dy); // d²f/dy²
      }
      if (std::abs(dz) > h) {
        hessian[5] += (f_j - f_i) / (dz * dz); // d²f/dz²
      }
    }
  }

  return hessian;
}

void AnisotropicErrorEstimator::apply_gradation(
    const MeshBase& mesh,
    std::vector<MetricTensor>& metrics) const {

  // Apply gradation control to ensure smooth size variation
  bool changed = true;
  size_t iterations = 0;

  while (changed && iterations < 10) {
    changed = false;

    for (size_t elem = 0; elem < metrics.size(); ++elem) {
      auto neighbors = mesh.get_element_neighbors(elem);

      for (size_t neighbor : neighbors) {
        // Check gradation between elements
        metrics[elem].compute_eigendecomposition();
        metrics[neighbor].compute_eigendecomposition();

        for (int i = 0; i < 3; ++i) {
          double ratio = metrics[elem].eigenvalues[i] / metrics[neighbor].eigenvalues[i];

          if (ratio > config_.gradation_parameter) {
            metrics[elem].eigenvalues[i] = metrics[neighbor].eigenvalues[i] *
                                            config_.gradation_parameter;
            changed = true;
          } else if (ratio < 1.0 / config_.gradation_parameter) {
            metrics[neighbor].eigenvalues[i] = metrics[elem].eigenvalues[i] *
                                                config_.gradation_parameter;
            changed = true;
          }
        }
      }
    }

    iterations++;
  }
}

void AnisotropicErrorEstimator::smooth_metric_field(
    const MeshBase& mesh,
    std::vector<MetricTensor>& metrics) const {

  // Smooth metric field using averaging
  std::vector<MetricTensor> smoothed = metrics;

  for (size_t elem = 0; elem < metrics.size(); ++elem) {
    auto neighbors = mesh.get_element_neighbors(elem);

    if (neighbors.empty()) continue;

    MetricTensor avg = metrics[elem];
    double weight = 1.0;

    for (size_t neighbor : neighbors) {
      avg = MetricTensor::interpolate(avg, metrics[neighbor], 0.5);
      weight += 1.0;
    }

    // Average with original
    smoothed[elem] = MetricTensor::interpolate(metrics[elem], avg, 0.5);
  }

  metrics = smoothed;
}

void AnisotropicErrorEstimator::normalize_metric_field(
    const MeshBase& mesh,
    std::vector<MetricTensor>& metrics) const {

  // Normalize metric field to achieve target complexity
  double current_complexity = 0.0;

  for (size_t elem = 0; elem < metrics.size(); ++elem) {
    metrics[elem].compute_eigendecomposition();

    // Complexity ~ sqrt(det(M))
    double det = 1.0;
    for (int i = 0; i < 3; ++i) {
      det *= metrics[elem].eigenvalues[i];
    }
    current_complexity += std::sqrt(det);
  }

  if (current_complexity > 0) {
    double scale = config_.target_elements / current_complexity;
    scale = std::pow(scale, 2.0 / mesh.get_dimension());

    for (auto& metric : metrics) {
      for (int i = 0; i < 3; ++i) {
        metric.eigenvalues[i] *= scale;
      }
    }
  }
}

//=============================================================================
// AnisotropicMarker Implementation
//=============================================================================

AnisotropicMarker::AnisotropicMarker(const Config& config)
    : config_(config) {}

void AnisotropicMarker::mark_elements(
    const MeshBase& mesh,
    const std::vector<double>& error_indicator,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {

  // Convert to anisotropic marks
  size_t num_elements = mesh.num_elements();
  marks.resize(num_elements, MarkType::NONE);

  // Simple threshold marking
  double max_error = *std::max_element(error_indicator.begin(), error_indicator.end());
  double threshold = max_error * 0.5;

  for (size_t elem = 0; elem < num_elements; ++elem) {
    if (error_indicator[elem] > threshold) {
      marks[elem] = MarkType::REFINE;
    }
  }
}

std::vector<AnisotropicMark> AnisotropicMarker::mark_anisotropic(
    const MeshBase& mesh,
    const std::vector<MetricTensor>& metric_field) const {

  size_t num_elements = mesh.num_elements();
  std::vector<AnisotropicMark> marks(num_elements);

  for (size_t elem = 0; elem < num_elements; ++elem) {
    marks[elem].element_id = elem;

    // Check metric conformity
    double conformity = compute_metric_conformity(mesh, elem, metric_field[elem]);

    if (conformity < config_.conformity_threshold) {
      marks[elem].base_mark = MarkType::REFINE;

      // Determine refinement directions
      marks[elem].refine_directions = determine_directions(mesh, elem, metric_field[elem]);

      // Set direction flags based on eigenvalues
      metric_field[elem].compute_eigendecomposition();

      for (int i = 0; i < 3; ++i) {
        if (metric_field[elem].eigenvalues[i] > config_.direction_threshold) {
          marks[elem].set_direction(i);
          marks[elem].direction_levels.push_back(1);
        }
      }
    }
  }

  // Add boundary layer marks if enabled
  if (config_.strategy == Config::Strategy::BOUNDARY_LAYER) {
    auto bl_marks = generate_boundary_layer_marks(mesh);
    marks.insert(marks.end(), bl_marks.begin(), bl_marks.end());
  }

  return marks;
}

double AnisotropicMarker::compute_metric_conformity(
    const MeshBase& mesh,
    size_t element_id,
    const MetricTensor& metric) const {

  // Compute how well element conforms to metric
  auto vertices = mesh.get_element_vertices(element_id);

  double conformity = 1.0;

  // Check edge lengths in metric
  for (size_t i = 0; i < vertices.size(); ++i) {
    for (size_t j = i + 1; j < vertices.size(); ++j) {
      std::array<double, 3> edge = {
        vertices[j][0] - vertices[i][0],
        vertices[j][1] - vertices[i][1],
        vertices[j][2] - vertices[i][2]
      };

      double metric_length = std::sqrt(metric.evaluate_in_direction(edge));
      double actual_length = std::sqrt(edge[0]*edge[0] + edge[1]*edge[1] + edge[2]*edge[2]);

      if (actual_length > 1e-12) {
        double ratio = metric_length / actual_length;
        conformity = std::min(conformity, std::min(ratio, 1.0/ratio));
      }
    }
  }

  return conformity;
}

std::vector<std::array<double, 3>> AnisotropicMarker::determine_directions(
    const MeshBase& mesh,
    size_t element_id,
    const MetricTensor& metric) const {

  std::vector<std::array<double, 3>> directions;

  MetricTensor m = metric;
  m.compute_eigendecomposition();

  // Use eigenvectors as refinement directions
  for (int i = 0; i < 3; ++i) {
    if (m.eigenvalues[i] > config_.direction_threshold) {
      directions.push_back(m.eigenvectors[i]);
    }
  }

  return directions;
}

bool AnisotropicMarker::needs_directional_refinement(
    const MeshBase& mesh,
    size_t element_id,
    const MetricTensor& metric) const {

  // Check if anisotropic refinement is beneficial
  MetricTensor m = metric;
  m.compute_eigendecomposition();

  return m.anisotropy_ratio > 2.0;
}

std::vector<AnisotropicMark> AnisotropicMarker::generate_boundary_layer_marks(
    const MeshBase& mesh) const {

  std::vector<AnisotropicMark> marks;

  // Find boundary elements
  for (size_t elem = 0; elem < mesh.num_elements(); ++elem) {
    if (mesh.is_boundary_element(elem)) {
      AnisotropicMark mark;
      mark.element_id = elem;
      mark.base_mark = MarkType::REFINE;

      // Compute normal direction
      auto normal = mesh.get_element_normal(elem);
      mark.refine_directions.push_back(normal);
      mark.set_direction(0);  // Refine in normal direction

      marks.push_back(mark);
    }
  }

  return marks;
}

//=============================================================================
// AnisotropicRefinementRules Implementation
//=============================================================================

AnisotropicRefinementRules::AnisotropicRefinementRules(const Config& config)
    : config_(config) {}

void AnisotropicRefinementRules::refine_anisotropic(
    MeshBase& mesh,
    const std::vector<AnisotropicMark>& marks) {

  for (const auto& mark : marks) {
    if (mark.base_mark != MarkType::REFINE) continue;

    auto elem_type = mesh.get_element_type(mark.element_id);

    switch (elem_type) {
      case ElementType::TRIANGLE:
        refine_triangle_directional(mesh, mark.element_id, mark);
        break;
      case ElementType::QUAD:
        refine_quad_directional(mesh, mark.element_id, mark);
        break;
      case ElementType::TETRAHEDRON:
        refine_tet_directional(mesh, mark.element_id, mark);
        break;
      case ElementType::HEXAHEDRON:
        refine_hex_directional(mesh, mark.element_id, mark);
        break;
      default:
        break;
    }
  }
}

RefinementPattern AnisotropicRefinementRules::get_anisotropic_pattern(
    ElementType type,
    const AnisotropicMark& mark) const {

  // Map anisotropic marks to refinement patterns
  if (type == ElementType::QUAD || type == ElementType::HEXAHEDRON) {
    if (mark.refine_in_direction(0) && !mark.refine_in_direction(1)) {
      return RefinementPattern::ANISOTROPIC_X;
    } else if (!mark.refine_in_direction(0) && mark.refine_in_direction(1)) {
      return RefinementPattern::ANISOTROPIC_Y;
    } else if (mark.refine_in_direction(0) && mark.refine_in_direction(1)) {
      return RefinementPattern::RED;
    }
  }

  return RefinementPattern::RED; // Default
}

void AnisotropicRefinementRules::refine_triangle_directional(
    MeshBase& mesh,
    size_t element_id,
    const AnisotropicMark& mark) {

  // Directional refinement for triangles
  auto vertices = mesh.get_element_vertex_ids(element_id);

  if (mark.refine_in_direction(0)) {
    // Bisect longest edge
    // Find longest edge and add midpoint
    // Create two new triangles
  }
}

void AnisotropicRefinementRules::refine_quad_directional(
    MeshBase& mesh,
    size_t element_id,
    const AnisotropicMark& mark) {

  auto vertices = mesh.get_element_vertex_ids(element_id);
  auto positions = mesh.get_element_vertices(element_id);

  if (mark.refine_in_direction(0) && !mark.refine_in_direction(1)) {
    // Refine in X direction only
    // Add midpoints on horizontal edges
    // Create two quads
  } else if (!mark.refine_in_direction(0) && mark.refine_in_direction(1)) {
    // Refine in Y direction only
    // Add midpoints on vertical edges
    // Create two quads
  }
}

void AnisotropicRefinementRules::refine_tet_directional(
    MeshBase& mesh,
    size_t element_id,
    const AnisotropicMark& mark) {
  // Directional refinement for tetrahedra
  // More complex - could use edge bisection or specialized patterns
}

void AnisotropicRefinementRules::refine_hex_directional(
    MeshBase& mesh,
    size_t element_id,
    const AnisotropicMark& mark) {
  // Directional refinement for hexahedra
  // Can refine in 1, 2, or 3 directions independently
}

//=============================================================================
// Simplified remaining implementations
//=============================================================================

// MetricMeshOptimizer
MetricMeshOptimizer::MetricMeshOptimizer(const Config& config) : config_(config) {}

void MetricMeshOptimizer::optimize(MeshBase& mesh, const std::vector<MetricTensor>& metric_field) {
  for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
    if (config_.enable_swapping) {
      edge_swapping_pass(mesh, metric_field);
    }
    if (config_.enable_smoothing) {
      vertex_smoothing_pass(mesh, metric_field);
    }
  }
}

// SizeField
SizeField::SizeField(const Config& config) : config_(config) {}

double SizeField::evaluate(const std::array<double, 3>& point) const {
  double size = config_.uniform_size;

  switch (config_.type) {
    case Type::UNIFORM:
      size = config_.uniform_size;
      break;
    case Type::ANALYTICAL:
      if (config_.size_function) {
        size = config_.size_function(point);
      }
      break;
    case Type::BACKGROUND:
      size = interpolate_from_background(point);
      break;
    default:
      break;
  }

  return apply_limits(size);
}

// BoundaryLayerGenerator
BoundaryLayerGenerator::BoundaryLayerGenerator(const Config& config) : config_(config) {}

void BoundaryLayerGenerator::generate_layers(MeshBase& mesh, const std::vector<int>& boundary_surfaces) {
  // Find boundary faces
  // Extrude to create layers
  // Create prism/hex elements
}

// AnisotropicAdaptivityManager
AnisotropicAdaptivityManager::AnisotropicAdaptivityManager(const Config& config)
    : config_(config) {
  error_estimator_ = std::make_unique<AnisotropicErrorEstimator>();
  marker_ = std::make_unique<AnisotropicMarker>();
  refinement_rules_ = std::make_unique<AnisotropicRefinementRules>();
  optimizer_ = std::make_unique<MetricMeshOptimizer>();
}

AdaptivityResult AnisotropicAdaptivityManager::adapt_anisotropic(
    MeshBase& mesh,
    MeshFields* fields,
    const AdaptivityOptions& options) {

  AdaptivityResult result;

  // Compute metric field
  if (config_.use_metric) {
    compute_metric_from_solution(mesh, fields);
  }

  // Mark elements
  auto aniso_marks = marker_->mark_anisotropic(mesh, metric_field_);

  // Apply refinement
  apply_refinement(mesh, aniso_marks);

  // Optimize mesh
  if (config_.optimize_mesh) {
    optimize_to_metric(mesh);
  }

  result.num_refined = aniso_marks.size();
  result.adaptation_complete = true;

  return result;
}

} // namespace svmp