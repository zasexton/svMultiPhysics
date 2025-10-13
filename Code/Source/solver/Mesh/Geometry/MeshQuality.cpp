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

#include "MeshQuality.h"
#include "../Core/MeshBase.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace svmp {

// Convert metric enum to string
std::string MeshQuality::metric_name(Metric m) {
  switch (m) {
    case Metric::AspectRatio: return "aspect_ratio";
    case Metric::Skewness: return "skewness";
    case Metric::Jacobian: return "jacobian";
    case Metric::EdgeRatio: return "edge_ratio";
    case Metric::MinAngle: return "min_angle";
    case Metric::MaxAngle: return "max_angle";
    case Metric::Warpage: return "warpage";
    case Metric::Taper: return "taper";
    case Metric::Stretch: return "stretch";
    case Metric::DiagonalRatio: return "diagonal_ratio";
    case Metric::ConditionNumber: return "condition_number";
    case Metric::ScaledJacobian: return "scaled_jacobian";
    case Metric::ShapeIndex: return "shape_index";
    case Metric::RelativeSizeSquared: return "relative_size_squared";
    case Metric::ShapeAndSize: return "shape_and_size";
    default: return "unknown";
  }
}

// Convert string to metric enum
MeshQuality::Metric MeshQuality::metric_from_name(const std::string& name) {
  if (name == "aspect_ratio") return Metric::AspectRatio;
  if (name == "skewness") return Metric::Skewness;
  if (name == "jacobian") return Metric::Jacobian;
  if (name == "edge_ratio") return Metric::EdgeRatio;
  if (name == "min_angle") return Metric::MinAngle;
  if (name == "max_angle") return Metric::MaxAngle;
  if (name == "warpage") return Metric::Warpage;
  if (name == "taper") return Metric::Taper;
  if (name == "stretch") return Metric::Stretch;
  if (name == "diagonal_ratio") return Metric::DiagonalRatio;
  if (name == "condition_number") return Metric::ConditionNumber;
  if (name == "scaled_jacobian") return Metric::ScaledJacobian;
  if (name == "shape_index") return Metric::ShapeIndex;
  if (name == "relative_size_squared") return Metric::RelativeSizeSquared;
  if (name == "shape_and_size") return Metric::ShapeAndSize;
  throw std::invalid_argument("Unknown quality metric: " + name);
}

// Main interface: compute quality for a single cell
real_t MeshQuality::compute(const MeshBase& mesh, index_t cell, Metric metric, Configuration cfg) {
  switch (metric) {
    case Metric::AspectRatio:
      return compute_aspect_ratio(mesh, cell, cfg);
    case Metric::Skewness:
      return compute_skewness(mesh, cell, cfg);
    case Metric::Jacobian:
      return compute_jacobian_quality(mesh, cell, cfg);
    case Metric::EdgeRatio:
      return compute_edge_ratio(mesh, cell, cfg);
    case Metric::MinAngle:
      return compute_min_angle(mesh, cell, cfg);
    case Metric::MaxAngle:
      return compute_max_angle(mesh, cell, cfg);
    case Metric::Warpage:
      return compute_warpage(mesh, cell, cfg);
    case Metric::Taper:
      return compute_taper(mesh, cell, cfg);
    case Metric::Stretch:
      return compute_stretch(mesh, cell, cfg);
    case Metric::DiagonalRatio:
      return compute_diagonal_ratio(mesh, cell, cfg);
    case Metric::ConditionNumber:
      return compute_condition_number(mesh, cell, cfg);
    case Metric::ScaledJacobian:
      return compute_scaled_jacobian(mesh, cell, cfg);
    case Metric::ShapeIndex:
      return compute_shape_index(mesh, cell, cfg);
    case Metric::RelativeSizeSquared:
      return compute_relative_size_squared(mesh, cell, cfg);
    case Metric::ShapeAndSize:
      return compute_shape_and_size(mesh, cell, cfg);
    default:
      return 0.0;
  }
}

// Compute quality for a single cell by metric name
real_t MeshQuality::compute(const MeshBase& mesh, index_t cell, const std::string& metric_str,
                           Configuration cfg) {
  Metric metric = metric_from_name(metric_str);
  return compute(mesh, cell, metric, cfg);
}

// Compute quality for all cells
std::vector<real_t> MeshQuality::compute_all(const MeshBase& mesh, Metric metric, Configuration cfg) {
  std::vector<real_t> qualities;
  qualities.reserve(mesh.n_cells());
  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    qualities.push_back(compute(mesh, static_cast<index_t>(c), metric, cfg));
  }
  return qualities;
}

// Get global min/max quality
std::pair<real_t,real_t> MeshQuality::global_range(const MeshBase& mesh, Metric metric,
                                                  Configuration cfg) {
  real_t min_quality = 1e300;
  real_t max_quality = -1e300;
  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    real_t q = compute(mesh, static_cast<index_t>(c), metric, cfg);
    min_quality = std::min(min_quality, q);
    max_quality = std::max(max_quality, q);
  }
  return {min_quality, max_quality};
}

// Get statistics for quality metric
MeshQuality::QualityStats MeshQuality::compute_statistics(const MeshBase& mesh, Metric metric,
                                                         Configuration cfg) {
  QualityStats stats;
  std::vector<real_t> qualities = compute_all(mesh, metric, cfg);

  if (qualities.empty()) return stats;

  // Basic statistics
  stats.min = *std::min_element(qualities.begin(), qualities.end());
  stats.max = *std::max_element(qualities.begin(), qualities.end());
  stats.mean = std::accumulate(qualities.begin(), qualities.end(), 0.0) / qualities.size();

  // Standard deviation
  real_t sq_sum = 0;
  for (real_t q : qualities) {
    sq_sum += (q - stats.mean) * (q - stats.mean);
  }
  stats.std_dev = std::sqrt(sq_sum / qualities.size());

  // Count by thresholds
  auto thresholds = get_thresholds(metric, mesh.cell_shape(0).family);
  for (real_t q : qualities) {
    if (q < thresholds.acceptable) stats.count_poor++;
    else if (q >= thresholds.good) stats.count_good++;
    if (q >= thresholds.excellent) stats.count_excellent++;
  }

  return stats;
}

// Quality thresholds for different metrics
MeshQuality::QualityThresholds MeshQuality::get_thresholds(Metric metric, CellFamily family) {
  QualityThresholds thresh;

  // These are example thresholds - should be refined based on application
  switch (metric) {
    case Metric::AspectRatio:
      // Lower is better for aspect ratio
      thresh.poor = 10.0;
      thresh.acceptable = 5.0;
      thresh.good = 2.0;
      thresh.excellent = 1.5;
      break;

    case Metric::Skewness:
      // Lower is better for skewness
      thresh.poor = 0.9;
      thresh.acceptable = 0.6;
      thresh.good = 0.3;
      thresh.excellent = 0.1;
      break;

    case Metric::MinAngle:
      // Higher is better for min angle
      if (family == CellFamily::Triangle) {
        thresh.poor = 10.0;
        thresh.acceptable = 20.0;
        thresh.good = 30.0;
        thresh.excellent = 40.0;
      } else if (family == CellFamily::Quad) {
        thresh.poor = 30.0;
        thresh.acceptable = 45.0;
        thresh.good = 60.0;
        thresh.excellent = 75.0;
      } else {
        // 3D elements
        thresh.poor = 15.0;
        thresh.acceptable = 25.0;
        thresh.good = 35.0;
        thresh.excellent = 45.0;
      }
      break;

    default:
      // Generic thresholds
      thresh.poor = 0.1;
      thresh.acceptable = 0.3;
      thresh.good = 0.6;
      thresh.excellent = 0.9;
  }

  return thresh;
}

// Check if cell quality is acceptable
bool MeshQuality::is_acceptable(const MeshBase& mesh, index_t cell, Metric metric, Configuration cfg) {
  real_t quality = compute(mesh, cell, metric, cfg);
  auto thresh = get_thresholds(metric, mesh.cell_shape(cell).family);

  // For metrics where lower is better (aspect ratio, skewness)
  if (metric == Metric::AspectRatio || metric == Metric::Skewness ||
      metric == Metric::MaxAngle || metric == Metric::Warpage) {
    return quality <= thresh.acceptable;
  }
  // For metrics where higher is better
  return quality >= thresh.acceptable;
}

// Find cells with poor quality
std::vector<index_t> MeshQuality::find_poor_quality_cells(const MeshBase& mesh, Metric metric,
                                                         real_t threshold, Configuration cfg) {
  std::vector<index_t> poor_cells;

  // Determine if metric is "higher is better" or "lower is better"
  bool higher_is_better = !(metric == Metric::AspectRatio || metric == Metric::Skewness ||
                           metric == Metric::MaxAngle || metric == Metric::Warpage);

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    real_t quality = compute(mesh, static_cast<index_t>(c), metric, cfg);
    bool is_poor = higher_is_better ? (quality < threshold) : (quality > threshold);
    if (is_poor) {
      poor_cells.push_back(static_cast<index_t>(c));
    }
  }

  return poor_cells;
}

// Helper: get cell vertices
std::vector<std::array<real_t,3>> MeshQuality::get_cell_vertices(const MeshBase& mesh, index_t cell,
                                                                 Configuration cfg) {
  auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(cell);
  std::vector<std::array<real_t,3>> vertices;
  vertices.reserve(n_nodes);

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  for (size_t i = 0; i < n_nodes; ++i) {
    index_t node_id = nodes_ptr[i];
    std::array<real_t,3> pt = {{0, 0, 0}};
    for (int d = 0; d < spatial_dim; ++d) {
      pt[d] = coords[node_id * spatial_dim + d];
    }
    vertices.push_back(pt);
  }

  return vertices;
}

// Compute aspect ratio
real_t MeshQuality::compute_aspect_ratio(const MeshBase& mesh, index_t cell, Configuration cfg) {
  // Simple implementation using bounding box
  auto vertices = get_cell_vertices(mesh, cell, cfg);

  if (vertices.empty()) return 1e300;

  std::array<real_t,3> min_pt = {{1e300, 1e300, 1e300}};
  std::array<real_t,3> max_pt = {{-1e300, -1e300, -1e300}};

  for (const auto& v : vertices) {
    for (int d = 0; d < 3; ++d) {
      min_pt[d] = std::min(min_pt[d], v[d]);
      max_pt[d] = std::max(max_pt[d], v[d]);
    }
  }

  real_t min_len = 1e300;
  real_t max_len = -1e300;
  int spatial_dim = mesh.dim();

  for (int d = 0; d < spatial_dim; ++d) {
    real_t len = max_pt[d] - min_pt[d];
    if (len > 1e-12) {
      min_len = std::min(min_len, len);
      max_len = std::max(max_len, len);
    }
  }

  if (min_len < 1e-12) return 1e300;
  return max_len / min_len;
}

// Compute edge ratio
real_t MeshQuality::compute_edge_ratio(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto& shape = mesh.cell_shape(cell);

  auto edge_lengths = compute_edge_lengths(vertices, shape);
  if (edge_lengths.empty()) return 1.0;

  real_t min_len = *std::min_element(edge_lengths.begin(), edge_lengths.end());
  real_t max_len = *std::max_element(edge_lengths.begin(), edge_lengths.end());

  if (min_len < 1e-12) return 1e300;
  return max_len / min_len;
}

// Helper: compute edge lengths
std::vector<real_t> MeshQuality::compute_edge_lengths(const std::vector<std::array<real_t,3>>& vertices,
                                                     const CellShape& shape) {
  std::vector<real_t> lengths;

  // Define edges based on cell type
  std::vector<std::pair<int,int>> edges;

  switch (shape.family) {
    case CellFamily::Triangle:
      edges = {{0,1}, {1,2}, {2,0}};
      break;
    case CellFamily::Quad:
      edges = {{0,1}, {1,2}, {2,3}, {3,0}};
      break;
    case CellFamily::Tetra:
      edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
      break;
    case CellFamily::Hex:
      edges = {{0,1}, {1,2}, {2,3}, {3,0},  // bottom face
              {4,5}, {5,6}, {6,7}, {7,4},  // top face
              {0,4}, {1,5}, {2,6}, {3,7}}; // vertical edges
      break;
    default:
      // For other shapes, compute all pairwise distances
      for (size_t i = 0; i < vertices.size(); ++i) {
        for (size_t j = i+1; j < vertices.size(); ++j) {
          edges.push_back({static_cast<int>(i), static_cast<int>(j)});
        }
      }
  }

  // Compute edge lengths
  for (const auto& [i, j] : edges) {
    real_t dx = vertices[j][0] - vertices[i][0];
    real_t dy = vertices[j][1] - vertices[i][1];
    real_t dz = vertices[j][2] - vertices[i][2];
    lengths.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
  }

  return lengths;
}

// Placeholder implementations for other metrics
real_t MeshQuality::compute_skewness(const MeshBase& mesh, index_t cell, Configuration cfg) {
  // Placeholder - would compute deviation from ideal element
  return 0.0;
}

real_t MeshQuality::compute_jacobian_quality(const MeshBase& mesh, index_t cell, Configuration cfg) {
  // Placeholder - would compute Jacobian determinant quality
  return 1.0;
}

real_t MeshQuality::compute_min_angle(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto& shape = mesh.cell_shape(cell);

  std::vector<real_t> angles;
  if (shape.is_2d()) {
    angles = compute_angles_2d(vertices);
  } else {
    angles = compute_angles_3d(vertices, shape);
  }

  if (angles.empty()) return 0.0;
  return *std::min_element(angles.begin(), angles.end());
}

real_t MeshQuality::compute_max_angle(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto& shape = mesh.cell_shape(cell);

  std::vector<real_t> angles;
  if (shape.is_2d()) {
    angles = compute_angles_2d(vertices);
  } else {
    angles = compute_angles_3d(vertices, shape);
  }

  if (angles.empty()) return 180.0;
  return *std::max_element(angles.begin(), angles.end());
}

// Helper: compute angles for 2D elements
std::vector<real_t> MeshQuality::compute_angles_2d(const std::vector<std::array<real_t,3>>& vertices) {
  std::vector<real_t> angles;
  size_t n = vertices.size();

  for (size_t i = 0; i < n; ++i) {
    size_t prev = (i + n - 1) % n;
    size_t next = (i + 1) % n;

    // Vectors from vertex i to neighbors
    real_t v1x = vertices[prev][0] - vertices[i][0];
    real_t v1y = vertices[prev][1] - vertices[i][1];
    real_t v2x = vertices[next][0] - vertices[i][0];
    real_t v2y = vertices[next][1] - vertices[i][1];

    // Compute angle using dot product
    real_t dot = v1x * v2x + v1y * v2y;
    real_t len1 = std::sqrt(v1x * v1x + v1y * v1y);
    real_t len2 = std::sqrt(v2x * v2x + v2y * v2y);

    if (len1 > 1e-12 && len2 > 1e-12) {
      real_t cos_angle = dot / (len1 * len2);
      cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
      angles.push_back(std::acos(cos_angle) * 180.0 / M_PI);
    }
  }

  return angles;
}

// Placeholder for 3D angle computation
std::vector<real_t> MeshQuality::compute_angles_3d(const std::vector<std::array<real_t,3>>& vertices,
                                                  const CellShape& shape) {
  // Would compute solid angles at vertices for 3D elements
  std::vector<real_t> angles;
  return angles;
}

// Placeholder implementations for remaining metrics
real_t MeshQuality::compute_warpage(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 0.0;
}

real_t MeshQuality::compute_taper(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 0.0;
}

real_t MeshQuality::compute_stretch(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 1.0;
}

real_t MeshQuality::compute_diagonal_ratio(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 1.0;
}

real_t MeshQuality::compute_condition_number(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 1.0;
}

real_t MeshQuality::compute_scaled_jacobian(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 1.0;
}

real_t MeshQuality::compute_shape_index(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 1.0;
}

real_t MeshQuality::compute_relative_size_squared(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 1.0;
}

real_t MeshQuality::compute_shape_and_size(const MeshBase& mesh, index_t cell, Configuration cfg) {
  return 1.0;
}

} // namespace svmp