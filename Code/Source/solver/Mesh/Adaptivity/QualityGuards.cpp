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
#include "../MeshBase.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

namespace svmp {

namespace {

// Helper functions for geometric computations
double compute_distance(const std::array<double, 3>& p1,
                         const std::array<double, 3>& p2) {
  double dx = p2[0] - p1[0];
  double dy = p2[1] - p1[1];
  double dz = p2[2] - p1[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double compute_angle(const std::array<double, 3>& v1,
                     const std::array<double, 3>& v2) {
  double dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  double len1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
  double len2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);

  if (len1 < 1e-12 || len2 < 1e-12) return 0.0;

  double cosine = dot / (len1 * len2);
  cosine = std::max(-1.0, std::min(1.0, cosine));

  return std::acos(cosine) * 180.0 / M_PI;
}

std::array<double, 3> cross_product(const std::array<double, 3>& v1,
                                     const std::array<double, 3>& v2) {
  return {v1[1] * v2[2] - v1[2] * v2[1],
          v1[2] * v2[0] - v1[0] * v2[2],
          v1[0] * v2[1] - v1[1] * v2[0]};
}

double compute_volume(const std::vector<std::array<double, 3>>& vertices) {
  if (vertices.size() == 3) {
    // Triangle area
    auto v1 = std::array<double, 3>{vertices[1][0] - vertices[0][0],
                                     vertices[1][1] - vertices[0][1],
                                     vertices[1][2] - vertices[0][2]};
    auto v2 = std::array<double, 3>{vertices[2][0] - vertices[0][0],
                                     vertices[2][1] - vertices[0][1],
                                     vertices[2][2] - vertices[0][2]};
    auto cross = cross_product(v1, v2);
    return 0.5 * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                           cross[2] * cross[2]);
  } else if (vertices.size() == 4) {
    // Quad area or tet volume
    // Check if coplanar for quad
    auto v1 = std::array<double, 3>{vertices[1][0] - vertices[0][0],
                                     vertices[1][1] - vertices[0][1],
                                     vertices[1][2] - vertices[0][2]};
    auto v2 = std::array<double, 3>{vertices[2][0] - vertices[0][0],
                                     vertices[2][1] - vertices[0][1],
                                     vertices[2][2] - vertices[0][2]};
    auto v3 = std::array<double, 3>{vertices[3][0] - vertices[0][0],
                                     vertices[3][1] - vertices[0][1],
                                     vertices[3][2] - vertices[0][2]};

    // Compute volume (zero if coplanar)
    auto cross = cross_product(v2, v3);
    double vol = std::abs(v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2]) / 6.0;

    if (vol < 1e-12) {
      // Coplanar - compute quad area
      double area1 = compute_volume({vertices[0], vertices[1], vertices[2]});
      double area2 = compute_volume({vertices[0], vertices[2], vertices[3]});
      return area1 + area2;
    }
    return vol;
  }
  // For higher order elements, approximate
  return 1.0;
}

} // anonymous namespace

//=============================================================================
// GeometricQualityChecker Implementation
//=============================================================================

GeometricQualityChecker::GeometricQualityChecker(const Config& config)
    : config_(config) {}

ElementQuality GeometricQualityChecker::compute_element_quality(
    const MeshBase& mesh, size_t elem_id) const {

  // Get element type
  auto elem_type = mesh.get_element_type(elem_id);

  switch (elem_type) {
    case ElementType::TRIANGLE:
      return compute_triangle_quality(mesh, elem_id);
    case ElementType::QUAD:
      return compute_quad_quality(mesh, elem_id);
    case ElementType::TETRAHEDRON:
      return compute_tet_quality(mesh, elem_id);
    case ElementType::HEXAHEDRON:
      return compute_hex_quality(mesh, elem_id);
    default:
      // Generic quality for unsupported types
      ElementQuality quality;
      quality.element_id = elem_id;
      quality.overall_quality = 0.5;
      return quality;
  }
}

MeshQuality GeometricQualityChecker::compute_mesh_quality(
    const MeshBase& mesh, const QualityOptions& options) const {

  MeshQuality mesh_quality;
  mesh_quality.min_quality = 1.0;
  mesh_quality.max_quality = 0.0;
  mesh_quality.quality_histogram.resize(10, 0); // 10 bins

  double quality_sum = 0.0;
  size_t num_elements = mesh.num_elements();

  for (size_t i = 0; i < num_elements; ++i) {
    ElementQuality elem_quality = compute_element_quality(mesh, i);
    double q = elem_quality.overall_quality();

    // Update statistics
    mesh_quality.min_quality = std::min(mesh_quality.min_quality, q);
    mesh_quality.max_quality = std::max(mesh_quality.max_quality, q);
    quality_sum += q;

    // Update histogram
    int bin = static_cast<int>(q * 10);
    if (bin >= 10) bin = 9;
    if (bin < 0) bin = 0;
    mesh_quality.quality_histogram[bin]++;

    // Check for poor quality
    if (q < options.min_quality) {
      mesh_quality.num_poor_elements++;
      mesh_quality.failed_elements.insert(i);
      mesh_quality.worst_elements.push_back(elem_quality);
    }

    // Check for inverted elements
    if (elem_quality.inverted) {
      mesh_quality.num_inverted++;
    }
  }

  // Sort worst elements
  std::sort(mesh_quality.worst_elements.begin(),
            mesh_quality.worst_elements.end(),
            [](const ElementQuality& a, const ElementQuality& b) {
              return a.overall_quality() < b.overall_quality();
            });

  // Keep only worst 10
  if (mesh_quality.worst_elements.size() > 10) {
    mesh_quality.worst_elements.resize(10);
  }

  mesh_quality.avg_quality = quality_sum / num_elements;
  mesh_quality.acceptable = (mesh_quality.min_quality >= options.min_quality &&
                              mesh_quality.num_inverted == 0);

  // Generate suggestions
  if (mesh_quality.num_inverted > 0) {
    mesh_quality.suggestions.push_back(
        "Fix inverted elements through smoothing or remeshing");
  }
  if (mesh_quality.num_poor_elements > num_elements * 0.1) {
    mesh_quality.suggestions.push_back(
        "Consider global smoothing to improve overall quality");
  }
  if (mesh_quality.min_quality < 0.1) {
    mesh_quality.suggestions.push_back(
        "Critical: Very poor quality elements detected - consider local remeshing");
  }

  return mesh_quality;
}

bool GeometricQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {

  if (quality.inverted) return false;
  if (quality.overall_quality() < options.min_quality) return false;
  if (quality.aspect_ratio > options.max_aspect_ratio) return false;
  if (quality.skewness > options.max_skewness) return false;
  if (quality.min_angle < options.min_angle) return false;

  return true;
}

ElementQuality GeometricQualityChecker::compute_triangle_quality(
    const MeshBase& mesh, size_t elem_id) const {

  ElementQuality quality;
  quality.element_id = elem_id;

  // Get triangle vertices
  auto vertices = mesh.get_element_vertices(elem_id);
  if (vertices.size() != 3) return quality;

  // Compute edge lengths
  double edges[3];
  edges[0] = compute_distance(vertices[0], vertices[1]);
  edges[1] = compute_distance(vertices[1], vertices[2]);
  edges[2] = compute_distance(vertices[2], vertices[0]);

  // Compute aspect ratio
  double max_edge = *std::max_element(edges, edges + 3);
  double min_edge = *std::min_element(edges, edges + 3);
  quality.edge_ratio = max_edge / min_edge;
  quality.aspect_ratio = quality.edge_ratio;

  // Compute angles
  compute_angles(vertices, quality.min_angle, quality.max_angle);

  // Compute area
  quality.size = compute_volume(vertices);

  // Check for inversion
  auto v1 = std::array<double, 3>{vertices[1][0] - vertices[0][0],
                                   vertices[1][1] - vertices[0][1],
                                   vertices[1][2] - vertices[0][2]};
  auto v2 = std::array<double, 3>{vertices[2][0] - vertices[0][0],
                                   vertices[2][1] - vertices[0][1],
                                   vertices[2][2] - vertices[0][2]};
  auto cross = cross_product(v1, v2);
  quality.jacobian = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                               cross[2] * cross[2]);
  quality.inverted = (quality.jacobian < 0);

  // Compute shape quality (based on equilateral triangle)
  double perimeter = edges[0] + edges[1] + edges[2];
  quality.shape_quality = 4.0 * std::sqrt(3.0) * quality.size / (perimeter * perimeter);

  // Compute skewness
  double ideal_angle = 60.0;
  quality.skewness = std::max(
      std::abs(quality.max_angle - ideal_angle) / (180.0 - ideal_angle),
      std::abs(ideal_angle - quality.min_angle) / ideal_angle);

  return quality;
}

ElementQuality GeometricQualityChecker::compute_quad_quality(
    const MeshBase& mesh, size_t elem_id) const {

  ElementQuality quality;
  quality.element_id = elem_id;

  // Get quad vertices
  auto vertices = mesh.get_element_vertices(elem_id);
  if (vertices.size() != 4) return quality;

  // Compute edge lengths
  double edges[4];
  edges[0] = compute_distance(vertices[0], vertices[1]);
  edges[1] = compute_distance(vertices[1], vertices[2]);
  edges[2] = compute_distance(vertices[2], vertices[3]);
  edges[3] = compute_distance(vertices[3], vertices[0]);

  // Compute aspect ratio
  double max_edge = *std::max_element(edges, edges + 4);
  double min_edge = *std::min_element(edges, edges + 4);
  quality.edge_ratio = max_edge / min_edge;
  quality.aspect_ratio = quality.edge_ratio;

  // Compute angles
  compute_angles(vertices, quality.min_angle, quality.max_angle);

  // Compute area
  quality.size = compute_volume(vertices);

  // Check for inversion - check both triangles
  auto check_triangle = [](const std::array<double, 3>& v0,
                           const std::array<double, 3>& v1,
                           const std::array<double, 3>& v2) {
    auto a = std::array<double, 3>{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    auto b = std::array<double, 3>{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    auto cross = cross_product(a, b);
    return cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
  };

  double jac1 = check_triangle(vertices[0], vertices[1], vertices[2]);
  double jac2 = check_triangle(vertices[0], vertices[2], vertices[3]);
  quality.jacobian = std::min(jac1, jac2);
  quality.inverted = (quality.jacobian < 0);

  // Compute shape quality
  double diag1 = compute_distance(vertices[0], vertices[2]);
  double diag2 = compute_distance(vertices[1], vertices[3]);
  quality.distortion = std::abs(diag1 - diag2) / std::max(diag1, diag2);
  quality.shape_quality = 1.0 - quality.distortion;

  // Compute skewness
  double ideal_angle = 90.0;
  quality.skewness = std::max(
      std::abs(quality.max_angle - ideal_angle) / (180.0 - ideal_angle),
      std::abs(ideal_angle - quality.min_angle) / ideal_angle);

  return quality;
}

ElementQuality GeometricQualityChecker::compute_tet_quality(
    const MeshBase& mesh, size_t elem_id) const {

  ElementQuality quality;
  quality.element_id = elem_id;

  // Get tet vertices
  auto vertices = mesh.get_element_vertices(elem_id);
  if (vertices.size() != 4) return quality;

  // Compute edge lengths
  double edges[6];
  int e = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      edges[e++] = compute_distance(vertices[i], vertices[j]);
    }
  }

  // Compute aspect ratio
  double max_edge = *std::max_element(edges, edges + 6);
  double min_edge = *std::min_element(edges, edges + 6);
  quality.edge_ratio = max_edge / min_edge;
  quality.aspect_ratio = quality.edge_ratio;

  // Compute volume
  quality.size = compute_volume(vertices);

  // Compute Jacobian
  auto v1 = std::array<double, 3>{vertices[1][0] - vertices[0][0],
                                   vertices[1][1] - vertices[0][1],
                                   vertices[1][2] - vertices[0][2]};
  auto v2 = std::array<double, 3>{vertices[2][0] - vertices[0][0],
                                   vertices[2][1] - vertices[0][1],
                                   vertices[2][2] - vertices[0][2]};
  auto v3 = std::array<double, 3>{vertices[3][0] - vertices[0][0],
                                   vertices[3][1] - vertices[0][1],
                                   vertices[3][2] - vertices[0][2]};

  auto cross = cross_product(v2, v3);
  quality.jacobian = v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2];
  quality.inverted = (quality.jacobian < 0);

  // Compute shape quality (normalized volume)
  double edge_sum = std::accumulate(edges, edges + 6, 0.0);
  quality.shape_quality = 216.0 * std::sqrt(2.0) * quality.size /
                          (edge_sum * edge_sum * edge_sum);

  return quality;
}

ElementQuality GeometricQualityChecker::compute_hex_quality(
    const MeshBase& mesh, size_t elem_id) const {

  ElementQuality quality;
  quality.element_id = elem_id;

  // Simplified hex quality - just check Jacobian at corners
  auto vertices = mesh.get_element_vertices(elem_id);
  if (vertices.size() != 8) return quality;

  // Compute edge lengths
  double min_edge = std::numeric_limits<double>::max();
  double max_edge = 0.0;

  // Check all 12 edges of hex
  int edge_pairs[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
      {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
      {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
  };

  for (int i = 0; i < 12; ++i) {
    double edge_len = compute_distance(vertices[edge_pairs[i][0]],
                                        vertices[edge_pairs[i][1]]);
    min_edge = std::min(min_edge, edge_len);
    max_edge = std::max(max_edge, edge_len);
  }

  quality.edge_ratio = max_edge / min_edge;
  quality.aspect_ratio = quality.edge_ratio;

  // Simplified shape quality
  quality.shape_quality = 1.0 / quality.aspect_ratio;

  return quality;
}

double GeometricQualityChecker::compute_aspect_ratio(
    const std::vector<std::array<double, 3>>& vertices) const {

  // Compute bounding box aspect ratio
  std::array<double, 3> min_pt = vertices[0];
  std::array<double, 3> max_pt = vertices[0];

  for (const auto& v : vertices) {
    for (int i = 0; i < 3; ++i) {
      min_pt[i] = std::min(min_pt[i], v[i]);
      max_pt[i] = std::max(max_pt[i], v[i]);
    }
  }

  double dx = max_pt[0] - min_pt[0];
  double dy = max_pt[1] - min_pt[1];
  double dz = max_pt[2] - min_pt[2];

  double max_dim = std::max({dx, dy, dz});
  double min_dim = std::min({dx + 1e-12, dy + 1e-12, dz + 1e-12});

  return max_dim / min_dim;
}

double GeometricQualityChecker::compute_skewness(
    const std::vector<std::array<double, 3>>& vertices) const {

  // Simplified skewness based on centroid deviation
  std::array<double, 3> centroid = {0, 0, 0};
  for (const auto& v : vertices) {
    for (int i = 0; i < 3; ++i) {
      centroid[i] += v[i];
    }
  }
  for (int i = 0; i < 3; ++i) {
    centroid[i] /= vertices.size();
  }

  // Compute average distance to centroid
  double avg_dist = 0.0;
  double max_dist = 0.0;
  double min_dist = std::numeric_limits<double>::max();

  for (const auto& v : vertices) {
    double dist = compute_distance(v, centroid);
    avg_dist += dist;
    max_dist = std::max(max_dist, dist);
    min_dist = std::min(min_dist, dist);
  }
  avg_dist /= vertices.size();

  if (avg_dist < 1e-12) return 1.0;

  return (max_dist - min_dist) / (2.0 * avg_dist);
}

void GeometricQualityChecker::compute_angles(
    const std::vector<std::array<double, 3>>& vertices,
    double& min_angle, double& max_angle) const {

  min_angle = 180.0;
  max_angle = 0.0;

  if (vertices.size() == 3) {
    // Triangle angles
    for (int i = 0; i < 3; ++i) {
      int j = (i + 1) % 3;
      int k = (i + 2) % 3;

      std::array<double, 3> v1 = {vertices[j][0] - vertices[i][0],
                                   vertices[j][1] - vertices[i][1],
                                   vertices[j][2] - vertices[i][2]};
      std::array<double, 3> v2 = {vertices[k][0] - vertices[i][0],
                                   vertices[k][1] - vertices[i][1],
                                   vertices[k][2] - vertices[i][2]};

      double angle = compute_angle(v1, v2);
      min_angle = std::min(min_angle, angle);
      max_angle = std::max(max_angle, angle);
    }
  } else if (vertices.size() == 4) {
    // Quad angles
    for (int i = 0; i < 4; ++i) {
      int j = (i + 3) % 4; // Previous
      int k = (i + 1) % 4; // Next

      std::array<double, 3> v1 = {vertices[j][0] - vertices[i][0],
                                   vertices[j][1] - vertices[i][1],
                                   vertices[j][2] - vertices[i][2]};
      std::array<double, 3> v2 = {vertices[k][0] - vertices[i][0],
                                   vertices[k][1] - vertices[i][1],
                                   vertices[k][2] - vertices[i][2]};

      double angle = compute_angle(v1, v2);
      min_angle = std::min(min_angle, angle);
      max_angle = std::max(max_angle, angle);
    }
  }
}

//=============================================================================
// JacobianQualityChecker Implementation
//=============================================================================

JacobianQualityChecker::JacobianQualityChecker(const Config& config)
    : config_(config) {}

ElementQuality JacobianQualityChecker::compute_element_quality(
    const MeshBase& mesh, size_t elem_id) const {

  ElementQuality quality;
  quality.element_id = elem_id;

  auto vertices = mesh.get_element_vertices(elem_id);
  auto elem_type = mesh.get_element_type(elem_id);

  // Get sample points
  auto sample_points = get_sample_points(static_cast<size_t>(elem_type));

  // Compute Jacobian at each sample point
  double min_jac = std::numeric_limits<double>::max();
  double max_jac = -std::numeric_limits<double>::max();
  double sum_jac = 0.0;

  for (const auto& xi : sample_points) {
    double jac = compute_jacobian_at_point(vertices, xi);
    min_jac = std::min(min_jac, jac);
    max_jac = std::max(max_jac, jac);
    sum_jac += jac;
  }

  quality.jacobian = min_jac;
  quality.inverted = (min_jac < config_.zero_tolerance);

  // Compute scaled Jacobian quality
  if (config_.use_scaled && max_jac > config_.zero_tolerance) {
    quality.shape_quality = min_jac / max_jac;
  } else {
    quality.shape_quality = min_jac > 0 ? 1.0 : 0.0;
  }

  // Compute condition number if requested
  if (config_.check_condition) {
    quality.distortion = compute_condition_number(vertices);
  }

  return quality;
}

MeshQuality JacobianQualityChecker::compute_mesh_quality(
    const MeshBase& mesh, const QualityOptions& options) const {

  MeshQuality mesh_quality;
  size_t num_elements = mesh.num_elements();

  for (size_t i = 0; i < num_elements; ++i) {
    ElementQuality elem_quality = compute_element_quality(mesh, i);

    if (elem_quality.inverted) {
      mesh_quality.num_inverted++;
      mesh_quality.failed_elements.insert(i);
    }

    if (elem_quality.jacobian < options.min_jacobian) {
      mesh_quality.num_poor_elements++;
      mesh_quality.failed_elements.insert(i);
    }
  }

  mesh_quality.acceptable = (mesh_quality.num_inverted == 0);

  return mesh_quality;
}

bool JacobianQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {

  if (quality.inverted) return false;
  if (quality.jacobian < options.min_jacobian) return false;

  return true;
}

double JacobianQualityChecker::compute_jacobian_at_point(
    const std::vector<std::array<double, 3>>& vertices,
    const std::array<double, 3>& xi) const {

  // Simplified Jacobian computation
  // For a real implementation, would use shape function derivatives

  if (vertices.size() == 3) {
    // Triangle Jacobian
    auto v1 = std::array<double, 3>{vertices[1][0] - vertices[0][0],
                                     vertices[1][1] - vertices[0][1],
                                     vertices[1][2] - vertices[0][2]};
    auto v2 = std::array<double, 3>{vertices[2][0] - vertices[0][0],
                                     vertices[2][1] - vertices[0][1],
                                     vertices[2][2] - vertices[0][2]};
    auto cross = cross_product(v1, v2);
    return std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                     cross[2] * cross[2]);
  } else if (vertices.size() == 4) {
    // Tet Jacobian
    auto v1 = std::array<double, 3>{vertices[1][0] - vertices[0][0],
                                     vertices[1][1] - vertices[0][1],
                                     vertices[1][2] - vertices[0][2]};
    auto v2 = std::array<double, 3>{vertices[2][0] - vertices[0][0],
                                     vertices[2][1] - vertices[0][1],
                                     vertices[2][2] - vertices[0][2]};
    auto v3 = std::array<double, 3>{vertices[3][0] - vertices[0][0],
                                     vertices[3][1] - vertices[0][1],
                                     vertices[3][2] - vertices[0][2]};
    auto cross = cross_product(v2, v3);
    return v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2];
  }

  return 1.0;
}

std::vector<std::array<double, 3>> JacobianQualityChecker::get_sample_points(
    size_t elem_type) const {

  std::vector<std::array<double, 3>> points;

  if (config_.corners_only) {
    // Only check at corners
    switch (static_cast<ElementType>(elem_type)) {
      case ElementType::TRIANGLE:
        points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        break;
      case ElementType::QUAD:
        points = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        break;
      case ElementType::TETRAHEDRON:
        points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        break;
      case ElementType::HEXAHEDRON:
        points = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        break;
      default:
        points = {{0.5, 0.5, 0.5}}; // Center point
    }
  } else {
    // Use Gauss points
    switch (static_cast<ElementType>(elem_type)) {
      case ElementType::TRIANGLE:
        points = {{1.0/3.0, 1.0/3.0, 0},
                  {0.6, 0.2, 0},
                  {0.2, 0.6, 0},
                  {0.2, 0.2, 0}};
        break;
      case ElementType::QUAD: {
        double g = 1.0 / std::sqrt(3.0);
        points = {{-g, -g, 0}, {g, -g, 0}, {g, g, 0}, {-g, g, 0}};
        break;
      }
      default:
        // Default to center point
        points = {{0.5, 0.5, 0.5}};
    }
  }

  return points;
}

double JacobianQualityChecker::compute_condition_number(
    const std::vector<std::array<double, 3>>& vertices) const {

  // Simplified condition number based on edge length ratios
  double min_edge = std::numeric_limits<double>::max();
  double max_edge = 0.0;

  for (size_t i = 0; i < vertices.size(); ++i) {
    for (size_t j = i + 1; j < vertices.size(); ++j) {
      double edge_len = compute_distance(vertices[i], vertices[j]);
      min_edge = std::min(min_edge, edge_len);
      max_edge = std::max(max_edge, edge_len);
    }
  }

  if (min_edge < 1e-12) return std::numeric_limits<double>::max();

  return max_edge / min_edge;
}

//=============================================================================
// SizeQualityChecker Implementation
//=============================================================================

SizeQualityChecker::SizeQualityChecker(const Config& config)
    : config_(config) {}

ElementQuality SizeQualityChecker::compute_element_quality(
    const MeshBase& mesh, size_t elem_id) const {

  ElementQuality quality;
  quality.element_id = elem_id;

  // Compute element size
  auto vertices = mesh.get_element_vertices(elem_id);
  quality.size = compute_volume(vertices);

  // Compute size gradation
  double gradation = compute_size_gradation(mesh, elem_id);
  quality.distortion = gradation - 1.0; // 0 = perfect gradation

  // Compute anisotropy if requested
  if (config_.check_anisotropy) {
    quality.aspect_ratio = compute_anisotropy(vertices);
  }

  // Overall quality based on gradation
  quality.shape_quality = 1.0 / (1.0 + quality.distortion);

  return quality;
}

MeshQuality SizeQualityChecker::compute_mesh_quality(
    const MeshBase& mesh, const QualityOptions& options) const {

  MeshQuality mesh_quality;
  size_t num_elements = mesh.num_elements();

  for (size_t i = 0; i < num_elements; ++i) {
    ElementQuality elem_quality = compute_element_quality(mesh, i);

    double gradation = 1.0 + elem_quality.distortion;
    if (gradation > config_.max_size_ratio) {
      mesh_quality.num_poor_elements++;
      mesh_quality.failed_elements.insert(i);
    }

    if (config_.check_anisotropy &&
        elem_quality.aspect_ratio > config_.max_anisotropy) {
      mesh_quality.failed_elements.insert(i);
    }
  }

  mesh_quality.acceptable = (mesh_quality.num_poor_elements == 0);

  if (mesh_quality.num_poor_elements > 0) {
    mesh_quality.suggestions.push_back(
        "Improve size gradation through graded refinement");
  }

  return mesh_quality;
}

bool SizeQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {

  double gradation = 1.0 + quality.distortion;
  if (gradation > config_.max_size_ratio) return false;

  if (config_.check_anisotropy &&
      quality.aspect_ratio > config_.max_anisotropy) {
    return false;
  }

  return true;
}

double SizeQualityChecker::compute_size_gradation(
    const MeshBase& mesh, size_t elem_id) const {

  // Get element size
  auto vertices = mesh.get_element_vertices(elem_id);
  double elem_size = compute_volume(vertices);

  // Get neighbors
  auto neighbors = get_neighbors(mesh, elem_id);

  if (neighbors.empty()) return 1.0;

  // Compute max size ratio with neighbors
  double max_ratio = 1.0;

  for (size_t neighbor_id : neighbors) {
    auto neighbor_vertices = mesh.get_element_vertices(neighbor_id);
    double neighbor_size = compute_volume(neighbor_vertices);

    if (neighbor_size > 1e-12) {
      double ratio = std::max(elem_size / neighbor_size,
                               neighbor_size / elem_size);
      max_ratio = std::max(max_ratio, ratio);
    }
  }

  return max_ratio;
}

double SizeQualityChecker::compute_anisotropy(
    const std::vector<std::array<double, 3>>& vertices) const {

  // Use bounding box to estimate anisotropy
  std::array<double, 3> min_pt = vertices[0];
  std::array<double, 3> max_pt = vertices[0];

  for (const auto& v : vertices) {
    for (int i = 0; i < 3; ++i) {
      min_pt[i] = std::min(min_pt[i], v[i]);
      max_pt[i] = std::max(max_pt[i], v[i]);
    }
  }

  double dx = max_pt[0] - min_pt[0];
  double dy = max_pt[1] - min_pt[1];
  double dz = max_pt[2] - min_pt[2];

  double max_dim = std::max({dx, dy, dz});
  double min_dim = std::min({dx + 1e-12, dy + 1e-12, dz + 1e-12});

  return max_dim / min_dim;
}

std::vector<size_t> SizeQualityChecker::get_neighbors(
    const MeshBase& mesh, size_t elem_id) const {

  // Get elements that share a face/edge with this element
  std::vector<size_t> neighbors;

  auto elem_vertices = mesh.get_element_vertex_ids(elem_id);
  size_t num_elements = mesh.num_elements();

  for (size_t i = 0; i < num_elements; ++i) {
    if (i == elem_id) continue;

    auto other_vertices = mesh.get_element_vertex_ids(i);

    // Check for shared vertices (simplified neighbor detection)
    int shared_count = 0;
    for (size_t v1 : elem_vertices) {
      for (size_t v2 : other_vertices) {
        if (v1 == v2) {
          shared_count++;
          break;
        }
      }
    }

    // Elements sharing 2+ vertices (edge) in 2D or 3+ vertices (face) in 3D
    // are neighbors
    if (shared_count >= 2) {
      neighbors.push_back(i);
    }
  }

  return neighbors;
}

//=============================================================================
// CompositeQualityChecker Implementation
//=============================================================================

void CompositeQualityChecker::add_checker(
    std::unique_ptr<QualityChecker> checker, double weight) {
  checkers_.emplace_back(std::move(checker), weight);
}

ElementQuality CompositeQualityChecker::compute_element_quality(
    const MeshBase& mesh, size_t elem_id) const {

  std::vector<std::pair<ElementQuality, double>> qualities;

  for (const auto& [checker, weight] : checkers_) {
    ElementQuality q = checker->compute_element_quality(mesh, elem_id);
    qualities.emplace_back(q, weight);
  }

  return combine_qualities(qualities);
}

MeshQuality CompositeQualityChecker::compute_mesh_quality(
    const MeshBase& mesh, const QualityOptions& options) const {

  std::vector<std::pair<MeshQuality, double>> qualities;

  for (const auto& [checker, weight] : checkers_) {
    MeshQuality q = checker->compute_mesh_quality(mesh, options);
    qualities.emplace_back(q, weight);
  }

  return combine_mesh_qualities(qualities);
}

bool CompositeQualityChecker::check_element(
    const ElementQuality& quality,
    const QualityOptions& options) const {

  // All checkers must pass
  for (const auto& [checker, weight] : checkers_) {
    if (!checker->check_element(quality, options)) {
      return false;
    }
  }

  return true;
}

ElementQuality CompositeQualityChecker::combine_qualities(
    const std::vector<std::pair<ElementQuality, double>>& qualities) const {

  if (qualities.empty()) {
    return ElementQuality{};
  }

  ElementQuality combined = qualities[0].first;
  double total_weight = 0.0;

  // Weighted average of metrics
  combined.aspect_ratio = 0.0;
  combined.skewness = 0.0;
  combined.jacobian = std::numeric_limits<double>::max();
  combined.shape_quality = 0.0;

  for (const auto& [q, weight] : qualities) {
    combined.aspect_ratio += weight * q.aspect_ratio;
    combined.skewness += weight * q.skewness;
    combined.jacobian = std::min(combined.jacobian, q.jacobian);
    combined.shape_quality += weight * q.shape_quality;
    combined.inverted = combined.inverted || q.inverted;
    total_weight += weight;
  }

  if (total_weight > 0) {
    combined.aspect_ratio /= total_weight;
    combined.skewness /= total_weight;
    combined.shape_quality /= total_weight;
  }

  return combined;
}

MeshQuality CompositeQualityChecker::combine_mesh_qualities(
    const std::vector<std::pair<MeshQuality, double>>& qualities) const {

  if (qualities.empty()) {
    return MeshQuality{};
  }

  MeshQuality combined = qualities[0].first;

  // Combine statistics
  for (size_t i = 1; i < qualities.size(); ++i) {
    const auto& q = qualities[i].first;
    combined.min_quality = std::min(combined.min_quality, q.min_quality);
    combined.max_quality = std::max(combined.max_quality, q.max_quality);
    combined.num_poor_elements = std::max(combined.num_poor_elements,
                                           q.num_poor_elements);
    combined.num_inverted = std::max(combined.num_inverted, q.num_inverted);

    // Union of failed elements
    combined.failed_elements.insert(q.failed_elements.begin(),
                                     q.failed_elements.end());

    // Combine suggestions
    combined.suggestions.insert(combined.suggestions.end(),
                                q.suggestions.begin(),
                                q.suggestions.end());
  }

  combined.acceptable = (combined.num_inverted == 0 &&
                         combined.num_poor_elements == 0);

  return combined;
}

//=============================================================================
// QualitySmoother Implementation
//=============================================================================

QualitySmoother::QualitySmoother(const Config& config)
    : config_(config) {}

size_t QualitySmoother::smooth(
    MeshBase& mesh,
    const QualityChecker& checker,
    const QualityOptions& options) {

  // Find all elements that need smoothing
  std::set<size_t> poor_elements;
  size_t num_elements = mesh.num_elements();

  for (size_t i = 0; i < num_elements; ++i) {
    ElementQuality quality = checker.compute_element_quality(mesh, i);
    if (!checker.check_element(quality, options)) {
      poor_elements.insert(i);
    }
  }

  if (poor_elements.empty()) {
    return 0; // No smoothing needed
  }

  return smooth_elements(mesh, poor_elements, checker, options);
}

size_t QualitySmoother::smooth_elements(
    MeshBase& mesh,
    const std::set<size_t>& element_ids,
    const QualityChecker& checker,
    const QualityOptions& options) {

  // Find nodes to smooth
  auto nodes = find_smoothing_nodes(mesh, element_ids);

  size_t iterations = 0;
  double prev_quality = 0.0;

  for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
    // Perform smoothing step based on method
    switch (config_.method) {
      case Config::Method::LAPLACIAN:
        laplacian_smooth(mesh, nodes);
        break;
      case Config::Method::SMART_LAPLACIAN:
        smart_laplacian_smooth(mesh, nodes, checker);
        break;
      case Config::Method::OPTIMIZATION_BASED:
        optimization_smooth(mesh, nodes, checker);
        break;
      case Config::Method::ANGLE_BASED:
        // Angle-based smoothing (not implemented)
        smart_laplacian_smooth(mesh, nodes, checker);
        break;
      case Config::Method::COMBINED:
        if (iter % 2 == 0) {
          smart_laplacian_smooth(mesh, nodes, checker);
        } else {
          optimization_smooth(mesh, nodes, checker);
        }
        break;
    }

    // Check convergence
    double total_quality = 0.0;
    for (size_t elem_id : element_ids) {
      ElementQuality q = checker.compute_element_quality(mesh, elem_id);
      total_quality += q.overall_quality();
    }

    if (iter > 0 && std::abs(total_quality - prev_quality) <
                     config_.convergence_tolerance) {
      break;
    }

    prev_quality = total_quality;
    iterations++;
  }

  return iterations;
}

void QualitySmoother::laplacian_smooth(
    MeshBase& mesh,
    const std::set<size_t>& nodes) {

  // Simple Laplacian smoothing
  std::map<size_t, std::array<double, 3>> new_positions;

  for (size_t node_id : nodes) {
    if (config_.preserve_boundary && is_boundary_node(mesh, node_id)) {
      continue;
    }

    // Get connected nodes
    auto neighbors = mesh.get_node_neighbors(node_id);
    if (neighbors.empty()) continue;

    // Compute average position
    std::array<double, 3> avg_pos = {0, 0, 0};
    for (size_t neighbor_id : neighbors) {
      auto pos = mesh.get_node_position(neighbor_id);
      for (int i = 0; i < 3; ++i) {
        avg_pos[i] += pos[i];
      }
    }

    for (int i = 0; i < 3; ++i) {
      avg_pos[i] /= neighbors.size();
    }

    // Apply relaxation
    auto current_pos = mesh.get_node_position(node_id);
    for (int i = 0; i < 3; ++i) {
      avg_pos[i] = current_pos[i] + config_.relaxation *
                                     (avg_pos[i] - current_pos[i]);
    }

    new_positions[node_id] = avg_pos;
  }

  // Update positions
  for (const auto& [node_id, pos] : new_positions) {
    mesh.set_node_position(node_id, pos);
  }
}

void QualitySmoother::smart_laplacian_smooth(
    MeshBase& mesh,
    const std::set<size_t>& nodes,
    const QualityChecker& checker) {

  // Smart Laplacian - only move if it improves quality
  std::map<size_t, std::array<double, 3>> new_positions;

  for (size_t node_id : nodes) {
    if (config_.preserve_boundary && is_boundary_node(mesh, node_id)) {
      continue;
    }

    // Get elements containing this node
    auto elements = mesh.get_node_elements(node_id);
    if (elements.empty()) continue;

    // Compute current quality
    double current_quality = 0.0;
    for (size_t elem_id : elements) {
      ElementQuality q = checker.compute_element_quality(mesh, elem_id);
      current_quality += q.overall_quality();
    }

    // Get connected nodes
    auto neighbors = mesh.get_node_neighbors(node_id);
    if (neighbors.empty()) continue;

    // Compute average position
    std::array<double, 3> avg_pos = {0, 0, 0};
    for (size_t neighbor_id : neighbors) {
      auto pos = mesh.get_node_position(neighbor_id);
      for (int i = 0; i < 3; ++i) {
        avg_pos[i] += pos[i];
      }
    }

    for (int i = 0; i < 3; ++i) {
      avg_pos[i] /= neighbors.size();
    }

    // Apply relaxation
    auto current_pos = mesh.get_node_position(node_id);
    for (int i = 0; i < 3; ++i) {
      avg_pos[i] = current_pos[i] + config_.relaxation *
                                     (avg_pos[i] - current_pos[i]);
    }

    // Test new position
    mesh.set_node_position(node_id, avg_pos);

    double new_quality = 0.0;
    bool has_inverted = false;
    for (size_t elem_id : elements) {
      ElementQuality q = checker.compute_element_quality(mesh, elem_id);
      new_quality += q.overall_quality();
      if (q.inverted) {
        has_inverted = true;
        break;
      }
    }

    // Accept or reject move
    if (!has_inverted && new_quality > current_quality) {
      new_positions[node_id] = avg_pos;
    }

    // Restore original position for now
    mesh.set_node_position(node_id, current_pos);
  }

  // Apply accepted moves
  for (const auto& [node_id, pos] : new_positions) {
    mesh.set_node_position(node_id, pos);
  }
}

void QualitySmoother::optimization_smooth(
    MeshBase& mesh,
    const std::set<size_t>& nodes,
    const QualityChecker& checker) {

  // Simple gradient-based optimization
  const double step_size = 0.01;
  const double epsilon = 1e-6;

  for (size_t node_id : nodes) {
    if (config_.preserve_boundary && is_boundary_node(mesh, node_id)) {
      continue;
    }

    // Get elements containing this node
    auto elements = mesh.get_node_elements(node_id);
    if (elements.empty()) continue;

    auto current_pos = mesh.get_node_position(node_id);

    // Compute gradient by finite differences
    std::array<double, 3> gradient = {0, 0, 0};

    for (int dim = 0; dim < 3; ++dim) {
      // Forward difference
      auto test_pos = current_pos;
      test_pos[dim] += epsilon;
      mesh.set_node_position(node_id, test_pos);

      double quality_plus = 0.0;
      for (size_t elem_id : elements) {
        ElementQuality q = checker.compute_element_quality(mesh, elem_id);
        quality_plus += q.overall_quality();
      }

      // Backward difference
      test_pos[dim] = current_pos[dim] - epsilon;
      mesh.set_node_position(node_id, test_pos);

      double quality_minus = 0.0;
      for (size_t elem_id : elements) {
        ElementQuality q = checker.compute_element_quality(mesh, elem_id);
        quality_minus += q.overall_quality();
      }

      gradient[dim] = (quality_plus - quality_minus) / (2.0 * epsilon);
    }

    // Gradient ascent step
    std::array<double, 3> new_pos = current_pos;
    for (int i = 0; i < 3; ++i) {
      new_pos[i] += step_size * gradient[i];
    }

    // Apply relaxation
    for (int i = 0; i < 3; ++i) {
      new_pos[i] = current_pos[i] + config_.relaxation *
                                     (new_pos[i] - current_pos[i]);
    }

    mesh.set_node_position(node_id, new_pos);

    // Check for inversion
    bool has_inverted = false;
    for (size_t elem_id : elements) {
      ElementQuality q = checker.compute_element_quality(mesh, elem_id);
      if (q.inverted) {
        has_inverted = true;
        break;
      }
    }

    // Revert if inverted
    if (has_inverted) {
      mesh.set_node_position(node_id, current_pos);
    }
  }
}

std::set<size_t> QualitySmoother::find_smoothing_nodes(
    const MeshBase& mesh,
    const std::set<size_t>& element_ids) const {

  std::set<size_t> nodes;

  for (size_t elem_id : element_ids) {
    auto elem_nodes = mesh.get_element_vertex_ids(elem_id);
    nodes.insert(elem_nodes.begin(), elem_nodes.end());
  }

  return nodes;
}

bool QualitySmoother::is_boundary_node(
    const MeshBase& mesh, size_t node_id) const {

  // Check if node is on mesh boundary
  return mesh.is_boundary_node(node_id);
}

bool QualitySmoother::is_feature_edge(
    const MeshBase& mesh, size_t v1, size_t v2) const {

  // Check if edge is a feature based on dihedral angle
  auto elements = mesh.get_edge_elements(v1, v2);
  if (elements.size() != 2) {
    return true; // Boundary edge
  }

  // Compute dihedral angle between elements
  // (simplified - would need proper normal computation)
  return false;
}

//=============================================================================
// QualityCheckerFactory Implementation
//=============================================================================

std::unique_ptr<QualityChecker> QualityCheckerFactory::create(
    const QualityOptions& options) {

  if (options.use_composite_checker) {
    return create_composite(options);
  }

  switch (options.primary_metric) {
    case QualityOptions::QualityMetric::ASPECT_RATIO:
      return create_geometric();

    case QualityOptions::QualityMetric::JACOBIAN:
      return create_jacobian();

    case QualityOptions::QualityMetric::SIZE_GRADATION:
      return create_size();

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

  auto composite = std::make_unique<CompositeQualityChecker>();

  // Add geometric checker
  composite->add_checker(create_geometric(), 1.0);

  // Add Jacobian checker if needed
  if (options.check_jacobian) {
    composite->add_checker(create_jacobian(), 0.5);
  }

  // Add size checker if gradation control is enabled
  if (options.enforce_gradation_control) {
    SizeQualityChecker::Config size_config;
    size_config.max_size_ratio = options.max_size_ratio;
    composite->add_checker(create_size(size_config), 0.5);
  }

  return composite;
}

//=============================================================================
// QualityGuardUtils Implementation
//=============================================================================

bool QualityGuardUtils::check_mesh_quality(
    const MeshBase& mesh,
    const QualityOptions& options) {

  auto checker = QualityCheckerFactory::create(options);
  MeshQuality quality = checker->compute_mesh_quality(mesh, options);

  return quality.acceptable;
}

std::set<size_t> QualityGuardUtils::find_poor_elements(
    const MeshBase& mesh,
    const QualityChecker& checker,
    const QualityOptions& options) {

  std::set<size_t> poor_elements;
  size_t num_elements = mesh.num_elements();

  for (size_t i = 0; i < num_elements; ++i) {
    ElementQuality quality = checker.compute_element_quality(mesh, i);
    if (!checker.check_element(quality, options)) {
      poor_elements.insert(i);
    }
  }

  return poor_elements;
}

double QualityGuardUtils::compute_quality_improvement(
    const MeshQuality& before,
    const MeshQuality& after) {

  double improvement = 0.0;

  // Average quality improvement
  improvement += (after.avg_quality - before.avg_quality);

  // Minimum quality improvement (weighted more)
  improvement += 2.0 * (after.min_quality - before.min_quality);

  // Reduction in poor elements
  if (before.num_poor_elements > 0) {
    double reduction = static_cast<double>(before.num_poor_elements -
                                            after.num_poor_elements) /
                       before.num_poor_elements;
    improvement += reduction;
  }

  // Reduction in inverted elements (critical)
  if (before.num_inverted > after.num_inverted) {
    improvement += 10.0; // High bonus for fixing inversions
  }

  return improvement;
}

void QualityGuardUtils::write_quality_report(
    const MeshQuality& quality,
    const std::string& filename) {

  std::ofstream file(filename);
  if (!file) return;

  file << "Mesh Quality Report\n";
  file << "==================\n\n";

  file << "Statistics:\n";
  file << "  Minimum quality: " << quality.min_quality << "\n";
  file << "  Maximum quality: " << quality.max_quality << "\n";
  file << "  Average quality: " << quality.avg_quality << "\n";
  file << "  Poor elements: " << quality.num_poor_elements << "\n";
  file << "  Inverted elements: " << quality.num_inverted << "\n";
  file << "  Acceptable: " << (quality.acceptable ? "Yes" : "No") << "\n\n";

  file << "Quality Histogram:\n";
  for (size_t i = 0; i < quality.quality_histogram.size(); ++i) {
    double range_start = i * 0.1;
    double range_end = (i + 1) * 0.1;
    file << "  [" << range_start << ", " << range_end << "): "
         << quality.quality_histogram[i] << "\n";
  }

  if (!quality.worst_elements.empty()) {
    file << "\nWorst Elements:\n";
    for (const auto& elem : quality.worst_elements) {
      file << "  Element " << elem.element_id
           << ": quality = " << elem.overall_quality()
           << ", aspect = " << elem.aspect_ratio
           << ", skew = " << elem.skewness << "\n";
    }
  }

  if (!quality.suggestions.empty()) {
    file << "\nSuggestions:\n";
    for (const auto& suggestion : quality.suggestions) {
      file << "  - " << suggestion << "\n";
    }
  }
}

std::vector<std::string> QualityGuardUtils::suggest_improvements(
    const MeshBase& mesh,
    const MeshQuality& quality) {

  std::vector<std::string> suggestions;

  if (quality.num_inverted > 0) {
    suggestions.push_back(
        "Critical: Fix " + std::to_string(quality.num_inverted) +
        " inverted elements");
  }

  if (quality.min_quality < 0.1) {
    suggestions.push_back(
        "Critical: Very poor quality elements detected (min = " +
        std::to_string(quality.min_quality) + ")");
  }

  if (quality.num_poor_elements > mesh.num_elements() * 0.2) {
    suggestions.push_back(
        "Warning: More than 20% of elements are poor quality");
  }

  if (quality.avg_quality < 0.5) {
    suggestions.push_back(
        "Consider global smoothing to improve average quality");
  }

  // Analyze histogram for specific issues
  size_t low_quality_count = 0;
  for (size_t i = 0; i < 3 && i < quality.quality_histogram.size(); ++i) {
    low_quality_count += quality.quality_histogram[i];
  }

  if (low_quality_count > mesh.num_elements() * 0.1) {
    suggestions.push_back(
        "Consider targeted refinement of low-quality regions");
  }

  return suggestions;
}

} // namespace svmp