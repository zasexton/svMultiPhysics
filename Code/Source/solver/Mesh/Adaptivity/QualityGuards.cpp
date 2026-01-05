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
#include <cmath>
#include <limits>
#include <numeric>

namespace svmp {

namespace {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

inline Vec3 to_vec3(const std::array<real_t, 3>& a) {
  return {static_cast<double>(a[0]), static_cast<double>(a[1]), static_cast<double>(a[2])};
}

inline Vec3 sub(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double norm(const Vec3& a) {
  return std::sqrt(dot(a, a));
}

inline double clamp01(double v) {
  return std::max(0.0, std::min(1.0, v));
}

inline double angle_deg(const Vec3& u, const Vec3& v) {
  const double nu = norm(u);
  const double nv = norm(v);
  if (nu <= 0.0 || nv <= 0.0) return 0.0;
  const double c = std::max(-1.0, std::min(1.0, dot(u, v) / (nu * nv)));
  constexpr double kPi = 3.14159265358979323846264338327950288;
  return std::acos(c) * 180.0 / kPi;
}

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
    report.quality_histogram.assign(10u, 0u);
    return report;
  }

  report.min_quality = 1.0;
  report.max_quality = 0.0;
  double sum = 0.0;
  report.quality_histogram.assign(10u, 0u);

  for (size_t i = 0; i < n; ++i) {
    const auto q = checker.compute_element_quality(mesh, i);
    const double s = q.overall_quality();
    report.min_quality = std::min(report.min_quality, s);
    report.max_quality = std::max(report.max_quality, s);
    sum += s;
    const size_t bin = std::min<size_t>(9u, static_cast<size_t>(std::floor(std::max(0.0, std::min(0.999999999999, s)) * 10.0)));
    report.quality_histogram[bin]++;
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
  ElementQuality q;
  q.element_id = elem_id;

  if (elem_id >= mesh.n_cells()) {
    return q;
  }

  const CellShape& shape = mesh.cell_shape(static_cast<index_t>(elem_id));
  const auto conn = mesh.cell_vertices(static_cast<index_t>(elem_id));
  const int num_corners = (shape.num_corners > 0) ? shape.num_corners : static_cast<int>(conn.size());
  if (num_corners <= 0 || static_cast<size_t>(num_corners) > conn.size()) {
    return q;
  }

  std::vector<Vec3> corners;
  corners.reserve(static_cast<size_t>(num_corners));
  for (int i = 0; i < num_corners; ++i) {
    corners.push_back(to_vec3(mesh.get_vertex_coords(conn[static_cast<size_t>(i)])));
  }

  auto edge_len = [&](int a, int b) {
    return norm(sub(corners[static_cast<size_t>(a)], corners[static_cast<size_t>(b)]));
  };

  double max_e = 0.0;
  double min_e = std::numeric_limits<double>::infinity();
  auto accum_edge = [&](int a, int b) {
    const double l = edge_len(a, b);
    max_e = std::max(max_e, l);
    min_e = std::min(min_e, l);
  };

  switch (shape.family) {
    case CellFamily::Line:
      if (num_corners >= 2) accum_edge(0, 1);
      break;
    case CellFamily::Triangle:
      if (num_corners >= 3) {
        accum_edge(0, 1);
        accum_edge(1, 2);
        accum_edge(2, 0);
      }
      break;
    case CellFamily::Quad:
      if (num_corners >= 4) {
        accum_edge(0, 1);
        accum_edge(1, 2);
        accum_edge(2, 3);
        accum_edge(3, 0);
      }
      break;
    case CellFamily::Tetra:
      if (num_corners >= 4) {
        accum_edge(0, 1);
        accum_edge(0, 2);
        accum_edge(0, 3);
        accum_edge(1, 2);
        accum_edge(1, 3);
        accum_edge(2, 3);
      }
      break;
    case CellFamily::Hex:
      if (num_corners >= 8) {
        const int e[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (const auto& ab : e) accum_edge(ab[0], ab[1]);
      }
      break;
    default:
      break;
  }

  if (!std::isfinite(min_e) || min_e <= 0.0) {
    q.aspect_ratio = 1e6;
    q.edge_ratio = q.aspect_ratio;
  } else {
    q.aspect_ratio = max_e / min_e;
    q.edge_ratio = q.aspect_ratio;
  }

  q.size = static_cast<double>(mesh.cell_measure(static_cast<index_t>(elem_id)));
  q.jacobian = q.size;

  if (shape.family == CellFamily::Triangle && num_corners == 3) {
    const Vec3 e01 = sub(corners[1], corners[0]);
    const Vec3 e02 = sub(corners[2], corners[0]);
    const double area = 0.5 * norm(cross(e01, e02));
    const double p = edge_len(0, 1) + edge_len(1, 2) + edge_len(2, 0);
    if (p > 0.0) {
      q.shape_quality = (4.0 * std::sqrt(3.0) * area) / (p * p);
    } else {
      q.shape_quality = 0.0;
    }
    q.shape_quality = clamp01(q.shape_quality);

    const double a0 = angle_deg(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
    const double a1 = angle_deg(sub(corners[0], corners[1]), sub(corners[2], corners[1]));
    const double a2 = angle_deg(sub(corners[0], corners[2]), sub(corners[1], corners[2]));
    q.min_angle = std::min({a0, a1, a2});
    q.max_angle = std::max({a0, a1, a2});
    q.skewness = std::max({std::abs(a0 - 60.0), std::abs(a1 - 60.0), std::abs(a2 - 60.0)}) / 60.0;
    q.skewness = clamp01(q.skewness);
    q.distortion = 0.0;
    q.inverted = false;
    return q;
  }

  if (shape.family == CellFamily::Quad && num_corners == 4) {
    const double a0 = angle_deg(sub(corners[1], corners[0]), sub(corners[3], corners[0]));
    const double a1 = angle_deg(sub(corners[2], corners[1]), sub(corners[0], corners[1]));
    const double a2 = angle_deg(sub(corners[3], corners[2]), sub(corners[1], corners[2]));
    const double a3 = angle_deg(sub(corners[0], corners[3]), sub(corners[2], corners[3]));
    q.min_angle = std::min({a0, a1, a2, a3});
    q.max_angle = std::max({a0, a1, a2, a3});
    q.skewness = std::max({std::abs(a0 - 90.0), std::abs(a1 - 90.0), std::abs(a2 - 90.0), std::abs(a3 - 90.0)}) / 90.0;
    q.skewness = clamp01(q.skewness);

    const double d1 = norm(sub(corners[2], corners[0]));
    const double d2 = norm(sub(corners[3], corners[1]));
    const double dmax = std::max(d1, d2);
    q.distortion = (dmax > 0.0) ? std::abs(d1 - d2) / dmax : 0.0;
    q.distortion = clamp01(q.distortion);
    q.shape_quality = clamp01(1.0 - q.distortion);
    q.inverted = false;
    return q;
  }

  // Coarse fallback for 3D elements: treat edge ratio as primary shape quality.
  q.min_angle = 60.0;
  q.max_angle = 90.0;
  q.skewness = 0.0;
  q.distortion = 0.0;
  q.shape_quality = clamp01((q.aspect_ratio > 0.0) ? (1.0 / q.aspect_ratio) : 0.0);
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
  ElementQuality q;
  q.element_id = elem_id;
  if (elem_id < mesh.n_cells()) {
    q.size = static_cast<double>(mesh.cell_measure(static_cast<index_t>(elem_id)));
    q.shape_quality = 1.0;
  }
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
