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
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace svmp {

GradientRecoveryEstimator::GradientRecoveryEstimator(const Config& config)
    : config_(config) {}

std::vector<double> GradientRecoveryEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {
  (void)fields;
  (void)options;

  const size_t n = mesh.n_cells();
  std::vector<double> indicators(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const double measure = mesh.cell_measure(static_cast<index_t>(i));
    indicators[i] = measure * measure;
  }
  return indicators;
}

std::vector<std::vector<double>> GradientRecoveryEstimator::recover_gradients(
    const MeshBase& mesh,
    const std::vector<double>& field_values) const {
  (void)field_values;
  return std::vector<std::vector<double>>(mesh.n_vertices(),
                                          std::vector<double>(static_cast<size_t>(mesh.dim()), 0.0));
}

double GradientRecoveryEstimator::compute_element_error(
    const MeshBase& mesh,
    size_t elem_id,
    const std::vector<double>& field_gradients,
    const std::vector<std::vector<double>>& recovered_gradients) const {
  (void)mesh;
  (void)elem_id;
  (void)field_gradients;
  (void)recovered_gradients;
  return 0.0;
}

JumpIndicatorEstimator::JumpIndicatorEstimator(const Config& config)
    : config_(config) {}

std::vector<double> JumpIndicatorEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {
  (void)fields;
  (void)options;

  const size_t n = mesh.n_cells();
  std::vector<double> indicators(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const double h = std::pow(mesh.cell_measure(static_cast<index_t>(i)),
                              1.0 / static_cast<double>(mesh.dim()));
    indicators[i] = h;
  }
  return indicators;
}

double JumpIndicatorEstimator::compute_face_jump(
    const MeshBase& mesh,
    size_t face_id,
    const std::vector<double>& field_values) const {
  (void)mesh;
  (void)face_id;
  (void)field_values;
  return 0.0;
}

ResidualBasedEstimator::ResidualBasedEstimator(const Config& config)
    : config_(config) {
  if (!config_.element_residual && config_.cell_residual) {
    config_.element_residual = config_.cell_residual;
  }
  if (!config_.edge_residual && config_.face_residual) {
    config_.edge_residual = config_.face_residual;
  }
  config_.include_edge_residuals = config_.include_edge_residuals || config_.include_face_residuals;
  if (!config_.element_residual) {
    throw std::invalid_argument("Element residual function must be provided");
  }
}

std::vector<double> ResidualBasedEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {
  (void)options;

  const size_t n = mesh.n_cells();
  std::vector<double> indicators(n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    double elem_residual = config_.element_residual(mesh, i, fields);
    if (config_.h_weighted) {
      const double h = std::pow(mesh.cell_measure(static_cast<index_t>(i)),
                                1.0 / static_cast<double>(mesh.dim()));
      elem_residual *= h * h;
    }
    indicators[i] = config_.scaling_constant * std::abs(elem_residual);
  }

  return indicators;
}

UserFieldEstimator::UserFieldEstimator(const Config& config)
    : config_(config) {}

std::vector<double> UserFieldEstimator::estimate(
    const MeshBase& mesh,
    const MeshFields* fields,
    const AdaptivityOptions& options) const {
  (void)fields;
  (void)options;

  if (!mesh.has_field(EntityKind::Volume, config_.error_field_name)) {
    throw std::runtime_error("Error field '" + config_.error_field_name + "' not found on cells");
  }
  if (mesh.field_type_by_name(EntityKind::Volume, config_.error_field_name) != FieldScalarType::Float64
      || mesh.field_components_by_name(EntityKind::Volume, config_.error_field_name) != 1) {
    throw std::runtime_error("Error field must be Float64 scalar cell field: '" + config_.error_field_name + "'");
  }

  const auto* data = static_cast<const double*>(
      mesh.field_data_by_name(EntityKind::Volume, config_.error_field_name));
  if (!data) {
    throw std::runtime_error("Failed to access error field storage");
  }

  const size_t n = mesh.n_cells();
  std::vector<double> indicators(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    indicators[i] = config_.scale_factor * data[i];
  }

  if (config_.normalize) {
    ErrorEstimatorUtils::normalize_indicators(indicators);
  }

  return indicators;
}

void MultiCriteriaEstimator::add_estimator(
    std::unique_ptr<ErrorEstimator> estimator,
    double weight) {
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

  std::vector<std::vector<double>> all_indicators;
  std::vector<double> weights;

  for (const auto& [estimator, weight] : estimators_) {
    all_indicators.push_back(estimator->estimate(mesh, fields, options));
    weights.push_back(weight);
  }

  return aggregate_indicators(all_indicators, weights);
}

bool MultiCriteriaEstimator::requires_fields() const {
  for (const auto& [estimator, weight] : estimators_) {
    (void)weight;
    if (estimator->requires_fields()) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> MultiCriteriaEstimator::required_field_names() const {
  std::vector<std::string> all_fields;
  for (const auto& [estimator, weight] : estimators_) {
    (void)weight;
    auto fields = estimator->required_field_names();
    all_fields.insert(all_fields.end(), fields.begin(), fields.end());
  }

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

  const size_t n = indicators.front().size();
  std::vector<double> result(n, 0.0);

  switch (aggregation_method_) {
    case AggregationMethod::WEIGHTED_SUM:
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < indicators.size(); ++j) {
          result[i] += weights[j] * indicators[j][i];
        }
      }
      break;

    case AggregationMethod::WEIGHTED_MAX:
      for (size_t i = 0; i < n; ++i) {
        double v = 0.0;
        for (size_t j = 0; j < indicators.size(); ++j) {
          v = std::max(v, weights[j] * indicators[j][i]);
        }
        result[i] = v;
      }
      break;

    case AggregationMethod::WEIGHTED_L2:
      for (size_t i = 0; i < n; ++i) {
        double sum_sq = 0.0;
        for (size_t j = 0; j < indicators.size(); ++j) {
          const double v = weights[j] * indicators[j][i];
          sum_sq += v * v;
        }
        result[i] = std::sqrt(sum_sq);
      }
      break;

    case AggregationMethod::WEIGHTED_LP:
      for (size_t i = 0; i < n; ++i) {
        double sum_p = 0.0;
        for (size_t j = 0; j < indicators.size(); ++j) {
          sum_p += std::pow(std::abs(weights[j] * indicators[j][i]), p_norm_);
        }
        result[i] = std::pow(sum_p, 1.0 / p_norm_);
      }
      break;
  }

  return result;
}

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
  }
  throw std::runtime_error("Unknown estimator type");
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

void ErrorEstimatorUtils::normalize_indicators(std::vector<double>& indicators) {
  if (indicators.empty()) {
    return;
  }
  const double max_val = *std::max_element(indicators.begin(), indicators.end());
  if (max_val > 0.0) {
    for (auto& v : indicators) {
      v /= max_val;
    }
  }
}

double ErrorEstimatorUtils::compute_global_error(
    const std::vector<double>& indicators,
    double p) {
  if (indicators.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const double v : indicators) {
    sum += std::pow(std::abs(v), p);
  }
  return std::pow(sum, 1.0 / p);
}

ErrorEstimatorUtils::ErrorStats ErrorEstimatorUtils::compute_statistics(
    const std::vector<double>& indicators) {
  ErrorStats stats{};
  if (indicators.empty()) {
    stats.min_error = 0.0;
    stats.max_error = 0.0;
    stats.mean_error = 0.0;
    stats.std_dev = 0.0;
    stats.total_error = 0.0;
    stats.num_cells = 0;
    stats.num_elements = 0;
    return stats;
  }

  stats.num_elements = indicators.size();
  stats.num_cells = stats.num_elements;
  stats.min_error = *std::min_element(indicators.begin(), indicators.end());
  stats.max_error = *std::max_element(indicators.begin(), indicators.end());
  stats.total_error = std::accumulate(indicators.begin(), indicators.end(), 0.0);
  stats.mean_error = stats.total_error / static_cast<double>(stats.num_elements);

  double sum_sq = 0.0;
  for (const double v : indicators) {
    const double d = v - stats.mean_error;
    sum_sq += d * d;
  }
  stats.std_dev = std::sqrt(sum_sq / static_cast<double>(stats.num_elements));
  return stats;
}

void ErrorEstimatorUtils::write_to_field(
    MeshBase& mesh,
    const std::string& field_name,
    const std::vector<double>& indicators) {
  if (!mesh.has_field(EntityKind::Volume, field_name)) {
    mesh.attach_field(EntityKind::Volume, field_name, FieldScalarType::Float64, 1);
  }
  if (mesh.field_type_by_name(EntityKind::Volume, field_name) != FieldScalarType::Float64
      || mesh.field_components_by_name(EntityKind::Volume, field_name) != 1) {
    throw std::runtime_error("Cannot write indicators into non-Float64 scalar cell field: '" + field_name + "'");
  }

  auto* data = static_cast<double*>(mesh.field_data_by_name(EntityKind::Volume, field_name));
  if (!data) {
    throw std::runtime_error("Failed to access indicator field storage");
  }
  const size_t n = std::min(indicators.size(), mesh.n_cells());
  for (size_t i = 0; i < n; ++i) {
    data[i] = indicators[i];
  }
}

} // namespace svmp
