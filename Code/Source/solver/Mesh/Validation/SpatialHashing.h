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

#ifndef SVMP_SPATIAL_HASHING_H
#define SVMP_SPATIAL_HASHING_H

#include "../Core/MeshBase.h"
#include "../Topology/CellShape.h"
// Note: GeometryKernels.h was removed - inlining necessary functions
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <iostream>

namespace svmp {
namespace validation {

// ====================
// P1.4: Spatial Hashing for Mesh Validation
// ====================
// Efficient O(N) duplicate vertex detection and inverted element checking
// using spatial hashing instead of O(N²) brute force comparisons.

// ---- Spatial hash grid for duplicate detection ----
class SpatialHashGrid {
public:
  explicit SpatialHashGrid(double tolerance = 1e-8)
    : tolerance_(tolerance), grid_size_(tolerance * 10.0) {}

  // Hash function for 3D coordinates
  struct GridKey {
    int64_t i, j, k;

    bool operator==(const GridKey& other) const {
      return i == other.i && j == other.j && k == other.k;
    }
  };

  struct GridKeyHash {
    std::size_t operator()(const GridKey& key) const {
      // Combine hash values using prime multipliers
      return std::hash<int64_t>()(key.i) * 73856093 +
             std::hash<int64_t>()(key.j) * 19349663 +
             std::hash<int64_t>()(key.k) * 83492791;
    }
  };

  // Add a point to the grid
  void insert(index_t id, const std::array<double, 3>& point) {
    auto key = get_grid_key(point);

    // Also check neighboring cells for tolerance
    for (int di = -1; di <= 1; ++di) {
      for (int dj = -1; dj <= 1; ++dj) {
        for (int dk = -1; dk <= 1; ++dk) {
          GridKey neighbor = {key.i + di, key.j + dj, key.k + dk};
          grid_[neighbor].push_back({id, point});
        }
      }
    }
  }

  // Find duplicate vertices within tolerance
  std::vector<std::pair<index_t, index_t>> find_duplicates() const {
    std::vector<std::pair<index_t, index_t>> duplicates;
    std::unordered_set<std::pair<index_t, index_t>, PairHash> seen;

    for (const auto& [key, points] : grid_) {
      // Check all pairs within this cell
      for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
          if (points[i].id == points[j].id) continue;

          double dist = distance(points[i].coord, points[j].coord);
          if (dist < tolerance_) {
            auto pair = std::make_pair(
              std::min(points[i].id, points[j].id),
              std::max(points[i].id, points[j].id)
            );
            if (seen.find(pair) == seen.end()) {
              duplicates.push_back(pair);
              seen.insert(pair);
            }
          }
        }
      }
    }

    return duplicates;
  }

  // Find vertices within tolerance of a given point
  std::vector<index_t> find_nearby(const std::array<double, 3>& point) const {
    std::vector<index_t> nearby;
    auto key = get_grid_key(point);

    // Check this cell and neighbors
    for (int di = -1; di <= 1; ++di) {
      for (int dj = -1; dj <= 1; ++dj) {
        for (int dk = -1; dk <= 1; ++dk) {
          GridKey neighbor = {key.i + di, key.j + dj, key.k + dk};
          auto it = grid_.find(neighbor);
          if (it != grid_.end()) {
            for (const auto& p : it->second) {
              if (distance(p.coord, point) < tolerance_) {
                nearby.push_back(p.id);
              }
            }
          }
        }
      }
    }

    return nearby;
  }

  void clear() {
    grid_.clear();
  }

  void set_tolerance(double tol) {
    tolerance_ = tol;
    grid_size_ = tol * 10.0;
    clear();
  }

private:
  struct Point {
    index_t id;
    std::array<double, 3> coord;
  };

  struct PairHash {
    std::size_t operator()(const std::pair<index_t, index_t>& p) const {
      return std::hash<index_t>()(p.first) ^ (std::hash<index_t>()(p.second) << 1);
    }
  };

  GridKey get_grid_key(const std::array<double, 3>& point) const {
    return {
      static_cast<int64_t>(std::floor(point[0] / grid_size_)),
      static_cast<int64_t>(std::floor(point[1] / grid_size_)),
      static_cast<int64_t>(std::floor(point[2] / grid_size_))
    };
  }

  double distance(const std::array<double, 3>& a, const std::array<double, 3>& b) const {
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
      double diff = a[i] - b[i];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

  double tolerance_;
  double grid_size_;
  std::unordered_map<GridKey, std::vector<Point>, GridKeyHash> grid_;
};

// ---- Mesh validation utilities using spatial hashing ----
class MeshValidator {
public:
  explicit MeshValidator(MeshBase& mesh, double tolerance = 1e-8)
    : mesh_(mesh), tolerance_(tolerance) {}

  // Find duplicate vertices in O(N) time
  std::vector<std::pair<index_t, index_t>> find_duplicate_vertices() {
    SpatialHashGrid grid(tolerance_);

    const auto& coords = mesh_.X_ref();
    int dim = mesh_.dim();
    size_t n_vertices = mesh_.n_vertices();

    // Insert all vertices into spatial hash grid
    for (size_t i = 0; i < n_vertices; ++i) {
      std::array<double, 3> pt = {0, 0, 0};
      for (int d = 0; d < dim; ++d) {
        pt[d] = coords[i * dim + d];
      }
      grid.insert(static_cast<index_t>(i), pt);
    }

    return grid.find_duplicates();
  }

  // Find inverted elements using robust Jacobian checks
  std::vector<index_t> find_inverted_elements() {
    std::vector<index_t> inverted;

    const auto& coords = mesh_.X_ref();
    int dim = mesh_.dim();
    size_t n_cells = mesh_.n_cells();

    for (size_t c = 0; c < n_cells; ++c) {
      auto [vertices_ptr, n_vertices] = mesh_.cell_vertices_span(static_cast<index_t>(c));
      const auto& shape = mesh_.cell_shape(static_cast<index_t>(c));

      // Get vertex coordinates
      std::vector<std::array<double, 3>> vertices(n_vertices);
      for (size_t i = 0; i < n_vertices; ++i) {
        for (int d = 0; d < dim; ++d) {
          vertices[i][d] = coords[vertices_ptr[i] * dim + d];
        }
        for (int d = dim; d < 3; ++d) {
          vertices[i][d] = 0.0;
        }
      }

      bool is_inverted = false;

      // Check based on cell type
      switch (shape.family) {
        case CellFamily::Triangle:
          if (n_vertices >= 3) {
            // Simple 2D orientation check for triangles
            double v1x = vertices[1][0] - vertices[0][0];
            double v1y = vertices[1][1] - vertices[0][1];
            double v2x = vertices[2][0] - vertices[0][0];
            double v2y = vertices[2][1] - vertices[0][1];
            double cross_z = v1x * v2y - v1y * v2x;
            is_inverted = (cross_z < 0);
          }
          break;
        case CellFamily::Tetra:
          if (n_vertices >= 4) {
            // Simple volume check for tetrahedra
            double v1x = vertices[1][0] - vertices[0][0];
            double v1y = vertices[1][1] - vertices[0][1];
            double v1z = vertices[1][2] - vertices[0][2];
            double v2x = vertices[2][0] - vertices[0][0];
            double v2y = vertices[2][1] - vertices[0][1];
            double v2z = vertices[2][2] - vertices[0][2];
            double v3x = vertices[3][0] - vertices[0][0];
            double v3y = vertices[3][1] - vertices[0][1];
            double v3z = vertices[3][2] - vertices[0][2];

            // Scalar triple product
            double volume = v1x * (v2y * v3z - v2z * v3y) -
                          v1y * (v2x * v3z - v2z * v3x) +
                          v1z * (v2x * v3y - v2y * v3x);
            is_inverted = (volume < 0);
          }
          break;
        case CellFamily::Hex:
          // Simplified check - just check if volume is negative
          if (n_vertices >= 8) {
            real_t measure = mesh_.cell_measure(static_cast<index_t>(c));
            is_inverted = (measure < 0);
          }
          break;
        default:
          // For other cell types, could implement additional checks
          break;
      }

      if (is_inverted) {
        inverted.push_back(static_cast<index_t>(c));
      }
    }

    return inverted;
  }

  // Find degenerate elements (zero or near-zero volume/area)
  std::vector<index_t> find_degenerate_elements(double volume_tol = 1e-10) {
    std::vector<index_t> degenerate;

    size_t n_cells = mesh_.n_cells();
    for (size_t c = 0; c < n_cells; ++c) {
      real_t measure = mesh_.cell_measure(static_cast<index_t>(c));
      if (std::abs(measure) < volume_tol) {
        degenerate.push_back(static_cast<index_t>(c));
      }
    }

    return degenerate;
  }

  // Find elements with poor quality (aspect ratio, skewness, etc.)
  std::vector<index_t> find_poor_quality_elements(const std::string& metric = "aspect_ratio",
                                                  double threshold = 10.0) {
    std::vector<index_t> poor_quality;

    size_t n_cells = mesh_.n_cells();
    for (size_t c = 0; c < n_cells; ++c) {
      real_t quality = mesh_.compute_quality(static_cast<index_t>(c), metric);
      if (quality > threshold) {
        poor_quality.push_back(static_cast<index_t>(c));
      }
    }

    return poor_quality;
  }

  // Comprehensive validation report
  void validate_and_report() {
    std::cout << "\n=== Mesh Validation Report ===" << std::endl;
    std::cout << "Mesh dimensions: " << mesh_.dim() << "D" << std::endl;
    std::cout << "Number of vertices: " << mesh_.n_vertices() << std::endl;
    std::cout << "Number of cells: " << mesh_.n_cells() << std::endl;
    std::cout << "Number of faces: " << mesh_.n_faces() << std::endl;

    // Check for duplicate vertices
    auto duplicates = find_duplicate_vertices();
    if (!duplicates.empty()) {
      std::cout << "\nWARNING: Found " << duplicates.size() << " duplicate vertex pairs:" << std::endl;
      for (size_t i = 0; i < std::min(size_t(10), duplicates.size()); ++i) {
        std::cout << "  Vertices " << duplicates[i].first << " and "
                  << duplicates[i].second << " are within tolerance" << std::endl;
      }
      if (duplicates.size() > 10) {
        std::cout << "  ... and " << (duplicates.size() - 10) << " more pairs" << std::endl;
      }
    } else {
      std::cout << "\n✓ No duplicate vertices found (tolerance: " << tolerance_ << ")" << std::endl;
    }

    // Check for inverted elements
    auto inverted = find_inverted_elements();
    if (!inverted.empty()) {
      std::cout << "\nERROR: Found " << inverted.size() << " inverted elements:" << std::endl;
      for (size_t i = 0; i < std::min(size_t(10), inverted.size()); ++i) {
        std::cout << "  Cell " << inverted[i] << " has negative Jacobian" << std::endl;
      }
      if (inverted.size() > 10) {
        std::cout << "  ... and " << (inverted.size() - 10) << " more inverted cells" << std::endl;
      }
    } else {
      std::cout << "✓ No inverted elements found" << std::endl;
    }

    // Check for degenerate elements
    auto degenerate = find_degenerate_elements();
    if (!degenerate.empty()) {
      std::cout << "\nWARNING: Found " << degenerate.size() << " degenerate elements:" << std::endl;
      for (size_t i = 0; i < std::min(size_t(5), degenerate.size()); ++i) {
        std::cout << "  Cell " << degenerate[i] << " has near-zero measure" << std::endl;
      }
      if (degenerate.size() > 5) {
        std::cout << "  ... and " << (degenerate.size() - 5) << " more degenerate cells" << std::endl;
      }
    } else {
      std::cout << "✓ No degenerate elements found" << std::endl;
    }

    // Quality statistics
    auto [min_quality, max_quality] = mesh_.global_quality_range("aspect_ratio");
    std::cout << "\nQuality metrics (aspect ratio):" << std::endl;
    std::cout << "  Minimum: " << min_quality << std::endl;
    std::cout << "  Maximum: " << max_quality << std::endl;

    auto poor_quality = find_poor_quality_elements("aspect_ratio", 10.0);
    if (!poor_quality.empty()) {
      std::cout << "  WARNING: " << poor_quality.size()
                << " elements have aspect ratio > 10.0" << std::endl;
    }

    std::cout << "\n=== End Validation Report ===" << std::endl;
  }

private:
  MeshBase& mesh_;
  double tolerance_;
};

} // namespace validation
} // namespace svmp

#endif // SVMP_SPATIAL_HASHING_H
