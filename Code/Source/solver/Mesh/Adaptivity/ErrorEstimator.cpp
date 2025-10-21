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

#include "ErrorEstimator.h"
#include "../Core/MeshBase.h"
#include "../Fields/MeshFields.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace svmp {

// ====================
// GradientRecoveryEstimator Implementation
// ====================

GradientRecoveryEstimator::GradientRecoveryEstimator(const Config& config)
    : config_(config) {
}

std::vector<double> GradientRecoveryEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {

  if (!fields) {
    throw std::runtime_error("GradientRecoveryEstimator requires field data");
  }

  // Get field values
  auto field_handle = fields->get_field(config_.field_name);
  if (!field_handle) {
    throw std::runtime_error("Field '" + config_.field_name + "' not found");
  }

  size_t num_elems = mesh.num_cells();
  std::vector<double> indicators(num_elems, 0.0);

  // TODO: Implement gradient recovery algorithm
  // 1. Compute gradients at element centers
  // 2. Recover smooth gradients at vertices using patch recovery
  // 3. Compute error as difference between computed and recovered gradients

  // Placeholder implementation
  for (size_t i = 0; i < num_elems; ++i) {
    // Simple placeholder: use element size as indicator
    double h = mesh.cell_measure(i);
    indicators[i] = h * h; // h^2 scaling for second-order accuracy
  }

  return indicators;
}

std::vector<std::vector<double>> GradientRecoveryEstimator::recover_gradients(
    const MeshBase& mesh,
    const std::vector<double>& field_values) const {

  size_t num_vertices = mesh.num_vertices();
  size_t dim = mesh.spatial_dim();

  std::vector<std::vector<double>> recovered(num_vertices, std::vector<double>(dim, 0.0));

  // TODO: Implement Zienkiewicz-Zhu gradient recovery
  // 1. For each vertex, identify surrounding patch of elements
  // 2. Fit polynomial to field values in patch
  // 3. Evaluate gradient of polynomial at vertex

  return recovered;
}

double GradientRecoveryEstimator::compute_element_error(
    const MeshBase& mesh,
    size_t elem_id,
    const std::vector<double>& field_gradients,
    const std::vector<std::vector<double>>& recovered_gradients) const {

  // TODO: Implement error computation
  // 1. Interpolate recovered gradients to element
  // 2. Compute L2 norm of difference
  // 3. Scale by element volume if requested

  return 0.0;
}

// ====================
// JumpIndicatorEstimator Implementation
// ====================

JumpIndicatorEstimator::JumpIndicatorEstimator(const Config& config)
    : config_(config) {
}

std::vector<double> JumpIndicatorEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {

  if (!fields) {
    throw std::runtime_error("JumpIndicatorEstimator requires field data");
  }

  auto field_handle = fields->get_field(config_.field_name);
  if (!field_handle) {
    throw std::runtime_error("Field '" + config_.field_name + "' not found");
  }

  size_t num_elems = mesh.num_cells();
  std::vector<double> indicators(num_elems, 0.0);

  // TODO: Implement jump indicator computation
  // 1. Loop over all interior faces
  // 2. Compute jump in field or derivative across face
  // 3. Accumulate contribution to adjacent elements

  // Placeholder implementation
  for (size_t i = 0; i < num_elems; ++i) {
    double h = std::pow(mesh.cell_measure(i), 1.0 / mesh.spatial_dim());
    indicators[i] = h; // h scaling for first-order jump
  }

  return indicators;
}

double JumpIndicatorEstimator::compute_face_jump(
    const MeshBase& mesh,
    size_t face_id,
    const std::vector<double>& field_values) const {

  // TODO: Implement face jump computation
  // 1. Get elements on both sides of face
  // 2. Compute field values/derivatives on both sides
  // 3. Compute jump based on configured type

  return 0.0;
}

// ====================
// ResidualBasedEstimator Implementation
// ====================

ResidualBasedEstimator::ResidualBasedEstimator(const Config& config)
    : config_(config) {

  if (!config_.element_residual) {
    throw std::invalid_argument("Element residual function must be provided");
  }
}

std::vector<double> ResidualBasedEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {

  size_t num_elems = mesh.num_cells();
  std::vector<double> indicators(num_elems, 0.0);

  for (size_t i = 0; i < num_elems; ++i) {
    // Compute element residual
    double elem_residual = config_.element_residual(mesh, i, fields);

    // Apply h-weighting if requested
    if (config_.h_weighted) {
      double h = std::pow(mesh.cell_measure(i), 1.0 / mesh.spatial_dim());
      elem_residual *= h * h; // h^2 weighting for elliptic problems
    }

    double total_residual = elem_residual * elem_residual;

    // Add edge residuals if requested
    if (config_.include_edge_residuals && config_.edge_residual) {
      // TODO: Loop over element edges/faces
      // double edge_residual = config_.edge_residual(mesh, edge_id, fields);
      // total_residual += edge_residual * edge_residual;
    }

    indicators[i] = config_.scaling_constant * std::sqrt(total_residual);
  }

  return indicators;
}

// ====================
// UserFieldEstimator Implementation
// ====================

UserFieldEstimator::UserFieldEstimator(const Config& config)
    : config_(config) {
}

std::vector<double> UserFieldEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {

  if (!fields) {
    throw std::runtime_error("UserFieldEstimator requires field data");
  }

  auto field_handle = fields->get_field(config_.error_field_name);
  if (!field_handle) {
    throw std::runtime_error("Error field '" + config_.error_field_name + "' not found");
  }

  size_t num_elems = mesh.num_cells();
  std::vector<double> indicators(num_elems);

  // Extract field values
  for (size_t i = 0; i < num_elems; ++i) {
    indicators[i] = field_handle->get_cell_value(i);

    // Apply scaling
    indicators[i] *= config_.scale_factor;
  }

  // Normalize if requested
  if (config_.normalize) {
    ErrorEstimatorUtils::normalize_indicators(indicators);
  }

  return indicators;
}

// ====================
// MultiCriteriaEstimator Implementation
// ====================

void MultiCriteriaEstimator::add_estimator(
    std::unique_ptr<ErrorEstimator> estimator, double weight) {

  if (!estimator) {
    throw std::invalid_argument("Cannot add null estimator");
  }
  if (weight < 0.0) {
    throw std::invalid_argument("Estimator weight must be non-negative");
  }

  estimators_.emplace_back(std::move(estimator), weight);
}

std::vector<double> MultiCriteriaEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {

  if (estimators_.empty()) {
    throw std::runtime_error("No estimators added to MultiCriteriaEstimator");
  }

  // Collect indicators from all estimators
  std::vector<std::vector<double>> all_indicators;
  std::vector<double> weights;

  for (const auto& [estimator, weight] : estimators_) {
    all_indicators.push_back(estimator->estimate(mesh, fields, options));
    weights.push_back(weight);
  }

  // Aggregate indicators
  return aggregate_indicators(all_indicators, weights);
}

bool MultiCriteriaEstimator::requires_fields() const {
  for (const auto& [estimator, weight] : estimators_) {
    if (estimator->requires_fields()) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> MultiCriteriaEstimator::required_field_names() const {
  std::vector<std::string> all_fields;

  for (const auto& [estimator, weight] : estimators_) {
    auto fields = estimator->required_field_names();
    all_fields.insert(all_fields.end(), fields.begin(), fields.end());
  }

  // Remove duplicates
  std::sort(all_fields.begin(), all_fields.end());
  all_fields.erase(std::unique(all_fields.begin(), all_fields.end()), all_fields.end());

  return all_fields;
}

std::vector<double> MultiCriteriaEstimator::aggregate_indicators(
    const std::vector<std::vector<double>>& indicators,
    const std::vector<double>& weights) const {

  if (indicators.empty()) {
    return {};
  }

  size_t num_elems = indicators[0].size();
  std::vector<double> result(num_elems, 0.0);

  switch (aggregation_method_) {
    case AggregationMethod::WEIGHTED_SUM:
      for (size_t i = 0; i < num_elems; ++i) {
        for (size_t j = 0; j < indicators.size(); ++j) {
          result[i] += weights[j] * indicators[j][i];
        }
      }
      break;

    case AggregationMethod::WEIGHTED_MAX:
      for (size_t i = 0; i < num_elems; ++i) {
        for (size_t j = 0; j < indicators.size(); ++j) {
          result[i] = std::max(result[i], weights[j] * indicators[j][i]);
        }
      }
      break;

    case AggregationMethod::WEIGHTED_L2:
      for (size_t i = 0; i < num_elems; ++i) {
        double sum_sq = 0.0;
        for (size_t j = 0; j < indicators.size(); ++j) {
          double val = weights[j] * indicators[j][i];
          sum_sq += val * val;
        }
        result[i] = std::sqrt(sum_sq);
      }
      break;

    case AggregationMethod::WEIGHTED_LP:
      for (size_t i = 0; i < num_elems; ++i) {
        double sum_p = 0.0;
        for (size_t j = 0; j < indicators.size(); ++j) {
          double val = weights[j] * indicators[j][i];
          sum_p += std::pow(std::abs(val), p_norm_);
        }
        result[i] = std::pow(sum_p, 1.0 / p_norm_);
      }
      break;
  }

  return result;
}

// ====================
// ErrorEstimatorFactory Implementation
// ====================

std::unique_ptr<ErrorEstimator> ErrorEstimatorFactory::create(
    const AdaptivityOptions& options) {

  switch (options.estimator_type) {
    case AdaptivityOptions::EstimatorType::GRADIENT_RECOVERY:
      return create_gradient_recovery();

    case AdaptivityOptions::EstimatorType::JUMP_INDICATOR:
      return create_jump_indicator();

    case AdaptivityOptions::EstimatorType::USER_FIELD:
      return create_user_field(options.user_field_name);

    case AdaptivityOptions::EstimatorType::MULTI_CRITERIA:
      return create_multi_criteria();

    case AdaptivityOptions::EstimatorType::RESIDUAL_BASED:
      throw std::runtime_error("Residual-based estimator requires custom configuration");

    default:
      throw std::runtime_error("Unknown estimator type");
  }
}

std::unique_ptr<ErrorEstimator> ErrorEstimatorFactory::create_gradient_recovery(
    const GradientRecoveryEstimator::Config& config) {
  return std::make_unique<GradientRecoveryEstimator>(config);
}

std::unique_ptr<ErrorEstimator> ErrorEstimatorFactory::create_jump_indicator(
    const JumpIndicatorEstimator::Config& config) {
  return std::make_unique<JumpIndicatorEstimator>(config);
}

std::unique_ptr<ErrorEstimator> ErrorEstimatorFactory::create_user_field(
    const std::string& field_name) {
  UserFieldEstimator::Config config;
  config.error_field_name = field_name;
  return std::make_unique<UserFieldEstimator>(config);
}

std::unique_ptr<MultiCriteriaEstimator> ErrorEstimatorFactory::create_multi_criteria() {
  return std::make_unique<MultiCriteriaEstimator>();
}

// ====================
// ErrorEstimatorUtils Implementation
// ====================

void ErrorEstimatorUtils::normalize_indicators(std::vector<double>& indicators) {
  if (indicators.empty()) return;

  // Find max value
  double max_val = *std::max_element(indicators.begin(), indicators.end());

  if (max_val > 0.0) {
    // Normalize to [0, 1]
    for (auto& val : indicators) {
      val /= max_val;
    }
  }
}

double ErrorEstimatorUtils::compute_global_error(
    const std::vector<double>& indicators, double p) {

  if (indicators.empty()) return 0.0;

  double sum = 0.0;
  for (double val : indicators) {
    sum += std::pow(std::abs(val), p);
  }

  return std::pow(sum, 1.0 / p);
}

ErrorEstimatorUtils::ErrorStats ErrorEstimatorUtils::compute_statistics(
    const std::vector<double>& indicators) {

  ErrorStats stats;

  if (indicators.empty()) {
    stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
    return stats;
  }

  stats.num_elements = indicators.size();
  stats.min_error = *std::min_element(indicators.begin(), indicators.end());
  stats.max_error = *std::max_element(indicators.begin(), indicators.end());

  // Mean
  stats.total_error = std::accumulate(indicators.begin(), indicators.end(), 0.0);
  stats.mean_error = stats.total_error / stats.num_elements;

  // Standard deviation
  double sum_sq_diff = 0.0;
  for (double val : indicators) {
    double diff = val - stats.mean_error;
    sum_sq_diff += diff * diff;
  }
  stats.std_dev = std::sqrt(sum_sq_diff / stats.num_elements);

  return stats;
}

void ErrorEstimatorUtils::write_to_field(
    MeshFields& fields,
    const std::string& field_name,
    const std::vector<double>& indicators) {

  // Create or get field
  auto field_handle = fields.get_field(field_name);
  if (!field_handle) {
    field_handle = fields.create_field(field_name, FieldLocation::CELL, 1);
  }

  // Write indicators
  for (size_t i = 0; i < indicators.size(); ++i) {
    field_handle->set_cell_value(i, indicators[i]);
  }
}

} // namespace svmp