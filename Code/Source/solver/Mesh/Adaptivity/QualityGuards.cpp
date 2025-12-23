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

#include "QualityGuards.h"
#include "../Core/MeshBase.h"
#include <algorithm>
#include <numeric>

namespace svmp {

namespace {

MeshQualityReport basic_mesh_quality_report(
    const MeshBase& mesh,
    const QualityChecker& checker,
    const QualityOptions& options) {
  MeshQualityReport report;
  const size_t n = mesh.n_cells();
  if (n == 0) {
    report.acceptable = true;
    report.min_quality = 1.0;
    report.max_quality = 1.0;
    report.avg_quality = 1.0;
    return report;
  }

  report.min_quality = 1.0;
  report.max_quality = 0.0;
  double sum = 0.0;

  for (size_t i = 0; i < n; ++i) {
    const auto q = checker.compute_element_quality(mesh, i);
    const double s = q.overall_quality();
    report.min_quality = std::min(report.min_quality, s);
    report.max_quality = std::max(report.max_quality, s);
    sum += s;
    if (s < options.min_quality_threshold) {
      report.failed_elements.insert(i);
      report.num_poor_elements++;
      report.num_poor_cells++;
    }
    if (q.inverted) {
      report.num_inverted++;
    }
  }

  report.avg_quality = sum / static_cast<double>(n);
  report.acceptable = report.failed_elements.empty();
  // Keep aliases consistent.
  report.num_poor_elements = report.num_poor_cells;
  return report;
}

} // namespace

GeometricQualityChecker::GeometricQualityChecker(const Config& config)
    : config_(config) {}

ElementQuality GeometricQualityChecker::compute_element_quality(
    const MeshBase& mesh,
    size_t elem_id) const {
  (void)mesh;
  ElementQuality q;
  q.element_id = elem_id;
  // Placeholder values; real computation is delegated to Mesh/Geometry/MeshQuality.
  q.aspect_ratio = 1.0;
  q.skewness = 0.0;
  q.jacobian = 1.0;
  q.min_angle = 60.0;
  q.max_angle = 90.0;
  q.size = 0.0;
  q.edge_ratio = 1.0;
  q.shape_quality = 1.0;
  q.distortion = 0.0;
  q.inverted = false;
  return q;
}

MeshQualityReport GeometricQualityChecker::compute_mesh_quality(
    const MeshBase& mesh,
    const QualityOptions& options) const {
  return basic_mesh_quality_report(mesh, *this, options);
}

bool GeometricQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {
  return quality.overall_quality() >= options.min_quality_threshold;
}

JacobianQualityChecker::JacobianQualityChecker(const Config& config)
    : config_(config) {}

ElementQuality JacobianQualityChecker::compute_element_quality(
    const MeshBase& mesh,
    size_t elem_id) const {
  (void)mesh;
  ElementQuality q;
  q.element_id = elem_id;
  q.jacobian = 1.0;
  q.inverted = false;
  return q;
}

MeshQualityReport JacobianQualityChecker::compute_mesh_quality(
    const MeshBase& mesh,
    const QualityOptions& options) const {
  return basic_mesh_quality_report(mesh, *this, options);
}

bool JacobianQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {
  return quality.overall_quality() >= options.min_quality_threshold;
}

SizeQualityChecker::SizeQualityChecker(const Config& config)
    : config_(config) {}

ElementQuality SizeQualityChecker::compute_element_quality(
    const MeshBase& mesh,
    size_t elem_id) const {
  (void)mesh;
  ElementQuality q;
  q.element_id = elem_id;
  q.size = 0.0;
  return q;
}

MeshQualityReport SizeQualityChecker::compute_mesh_quality(
    const MeshBase& mesh,
    const QualityOptions& options) const {
  return basic_mesh_quality_report(mesh, *this, options);
}

bool SizeQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {
  return quality.overall_quality() >= options.min_quality_threshold;
}

void CompositeQualityChecker::add_checker(
    std::unique_ptr<QualityChecker> checker,
    double weight) {
  if (!checker) {
    return;
  }
  checkers_.emplace_back(std::move(checker), weight);
}

ElementQuality CompositeQualityChecker::compute_element_quality(
    const MeshBase& mesh,
    size_t elem_id) const {
  if (checkers_.empty()) {
    GeometricQualityChecker::Config cfg;
    GeometricQualityChecker fallback(cfg);
    return fallback.compute_element_quality(mesh, elem_id);
  }
  return checkers_.front().first->compute_element_quality(mesh, elem_id);
}

MeshQualityReport CompositeQualityChecker::compute_mesh_quality(
    const MeshBase& mesh,
    const QualityOptions& options) const {
  return basic_mesh_quality_report(mesh, *this, options);
}

bool CompositeQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {
  return quality.overall_quality() >= options.min_quality_threshold;
}

ElementQuality CompositeQualityChecker::combine_qualities(
    const std::vector<std::pair<ElementQuality, double>>& qualities) const {
  if (qualities.empty()) {
    return {};
  }
  return qualities.front().first;
}

MeshQualityReport CompositeQualityChecker::combine_mesh_qualities(
    const std::vector<std::pair<MeshQualityReport, double>>& qualities) const {
  if (qualities.empty()) {
    return {};
  }
  return qualities.front().first;
}

QualitySmoother::QualitySmoother(const Config& config)
    : config_(config) {}

size_t QualitySmoother::smooth(
    MeshBase& mesh,
    const QualityChecker& checker,
    const QualityOptions& options) {
  (void)mesh;
  (void)checker;
  (void)options;
  return 0;
}

size_t QualitySmoother::smooth_elements(
    MeshBase& mesh,
    const std::set<size_t>& element_ids,
    const QualityChecker& checker,
    const QualityOptions& options) {
  (void)mesh;
  (void)element_ids;
  (void)checker;
  (void)options;
  return 0;
}

void QualitySmoother::laplacian_smooth(
    MeshBase& mesh,
    const std::set<size_t>& nodes) {
  (void)mesh;
  (void)nodes;
}

void QualitySmoother::smart_laplacian_smooth(
    MeshBase& mesh,
    const std::set<size_t>& nodes,
    const QualityChecker& checker) {
  (void)mesh;
  (void)nodes;
  (void)checker;
}

void QualitySmoother::optimization_smooth(
    MeshBase& mesh,
    const std::set<size_t>& nodes,
    const QualityChecker& checker) {
  (void)mesh;
  (void)nodes;
  (void)checker;
}

std::set<size_t> QualitySmoother::find_smoothing_nodes(
    const MeshBase& mesh,
    const std::set<size_t>& element_ids) const {
  (void)mesh;
  (void)element_ids;
  return {};
}

bool QualitySmoother::is_boundary_node(
    const MeshBase& mesh,
    size_t node_id) const {
  (void)mesh;
  (void)node_id;
  return false;
}

bool QualitySmoother::is_feature_edge(
    const MeshBase& mesh,
    size_t v1, size_t v2) const {
  (void)mesh;
  (void)v1;
  (void)v2;
  return false;
}

std::unique_ptr<QualityChecker> QualityCheckerFactory::create(const QualityOptions& options) {
  switch (options.primary_metric) {
    case QualityOptions::QualityMetric::JACOBIAN:
      return create_jacobian();
    case QualityOptions::QualityMetric::SIZE_GRADATION:
      return create_size();
    case QualityOptions::QualityMetric::ASPECT_RATIO:
    default:
      return create_geometric();
  }
}

std::unique_ptr<QualityChecker> QualityCheckerFactory::create_geometric(
    const GeometricQualityChecker::Config& config) {
  return std::make_unique<GeometricQualityChecker>(config);
}

std::unique_ptr<QualityChecker> QualityCheckerFactory::create_jacobian(
    const JacobianQualityChecker::Config& config) {
  return std::make_unique<JacobianQualityChecker>(config);
}

std::unique_ptr<QualityChecker> QualityCheckerFactory::create_size(
    const SizeQualityChecker::Config& config) {
  return std::make_unique<SizeQualityChecker>(config);
}

std::unique_ptr<QualityChecker> QualityCheckerFactory::create_composite(
    const QualityOptions& options) {
  (void)options;
  auto composite = std::make_unique<CompositeQualityChecker>();
  composite->add_checker(create_geometric(), 1.0);
  return composite;
}

bool QualityGuardUtils::check_mesh_quality(
    const MeshBase& mesh,
    const QualityOptions& options) {
  auto checker = QualityCheckerFactory::create(options);
  const auto report = checker->compute_mesh_quality(mesh, options);
  return report.acceptable;
}

std::set<size_t> QualityGuardUtils::find_poor_elements(
    const MeshBase& mesh,
    const QualityChecker& checker,
    const QualityOptions& options) {
  return checker.compute_mesh_quality(mesh, options).failed_elements;
}

double QualityGuardUtils::compute_quality_improvement(
    const MeshQualityReport& before,
    const MeshQualityReport& after) {
  return after.avg_quality - before.avg_quality;
}

void QualityGuardUtils::write_quality_report(
    const MeshQualityReport& quality,
    const std::string& filename) {
  (void)quality;
  (void)filename;
}

std::vector<std::string> QualityGuardUtils::suggest_improvements(
    const MeshBase& mesh,
    const MeshQualityReport& quality) {
  (void)mesh;
  (void)quality;
  return {};
}

} // namespace svmp
