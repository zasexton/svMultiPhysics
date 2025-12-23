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

#ifndef SVMP_ERROR_ESTIMATOR_DETAIL_H
#define SVMP_ERROR_ESTIMATOR_DETAIL_H

#include "../Core/MeshBase.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace svmp {
namespace detail {

/**
 * @brief Compute characteristic length for a cell
 *
 * Uses h = measure^(1/dim) for consistent dimensional scaling
 * This gives edge length for simplices, side length for cubes, etc.
 *
 * @param mesh The mesh
 * @param cell_id Cell index
 * @return Characteristic length h
 */
inline double compute_characteristic_length(const MeshBase& mesh, index_t cell_id) {
  size_t dim = mesh.dim();
  double measure = mesh.cell_measure(cell_id);
  // h = measure^(1/dim) for consistent dimensional scaling
  return std::pow(measure, 1.0 / dim);
}

/**
 * @brief Compute domain-scaled epsilon for numerical stability
 *
 * Returns epsilon = 1e-12 * domain_size where domain_size is the
 * maximum extent of the mesh bounding box. This prevents issues
 * with absolute tolerance values on different mesh scales.
 *
 * @param mesh The mesh
 * @return Domain-scaled epsilon
 */
inline double compute_domain_epsilon(const MeshBase& mesh) {
  auto [bbox_min, bbox_max] = mesh.bounding_box();
  double domain_size = 0.0;
  for (int d = 0; d < 3; ++d) {
    double extent = bbox_max[d] - bbox_min[d];
    domain_size = std::max(domain_size, extent);
  }
  return 1e-12 * domain_size;  // Scale epsilon by domain size
}

/**
 * @brief Compute cell gradient using weighted least-squares
 *
 * Builds a least-squares system using cell neighbors:
 *   minimize sum_i w_i * |grad·(x_i - x_c) - (u_i - u_c)|^2
 * where w_i = 1/|x_i - x_c|^2 (inverse distance squared weighting)
 *
 * This gives a robust gradient estimate that is O(1) accurate
 * for linear fields and converges as h^p for smooth fields.
 *
 * @param mesh The mesh
 * @param cell_id Cell index
 * @param field_values Cell-centered field values
 * @return Gradient vector (size = mesh.dim())
 */
inline std::vector<double> compute_cell_gradient_lsq(
    const MeshBase& mesh,
    index_t cell_id,
    const std::vector<double>& field_values) {

  size_t dim = mesh.dim();
  std::vector<double> gradient(dim, 0.0);

  // Get cell center and value
  auto center = mesh.cell_center(cell_id);
  double center_value = field_values[cell_id];

  // Get neighbors using adjacency (O(1) instead of O(N))
  auto neighbors = mesh.cell_neighbors(cell_id);

  if (neighbors.empty()) {
    return gradient;  // Isolated cell, zero gradient
  }

  // Build least-squares system: A^T A x = A^T b
  // where x is the gradient vector we're solving for
  std::vector<double> AtA(dim * dim, 0.0);
  std::vector<double> Atb(dim, 0.0);

  for (index_t neighbor_id : neighbors) {
    if (neighbor_id < 0 || neighbor_id >= static_cast<index_t>(mesh.n_cells())) {
      continue;  // Skip invalid neighbors
    }

    auto neighbor_center = mesh.cell_center(neighbor_id);
    double neighbor_value = field_values[neighbor_id];

    // Direction vector and distance
    std::vector<double> dir(dim);
    double dist_sq = 0.0;
    for (size_t d = 0; d < dim; ++d) {
      dir[d] = neighbor_center[d] - center[d];
      dist_sq += dir[d] * dir[d];
    }

    if (dist_sq < 1e-20) continue;  // Skip coincident cells

    double weight = 1.0 / dist_sq;  // Inverse distance weighting
    double value_diff = neighbor_value - center_value;

    // Add to normal equations
    for (size_t i = 0; i < dim; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        AtA[i * dim + j] += weight * dir[i] * dir[j];
      }
      Atb[i] += weight * dir[i] * value_diff;
    }
  }

  // Solve the system (simplified for 2D/3D)
  if (dim == 2) {
    double det = AtA[0] * AtA[3] - AtA[1] * AtA[2];
    if (std::abs(det) > 1e-20) {
      gradient[0] = (AtA[3] * Atb[0] - AtA[1] * Atb[1]) / det;
      gradient[1] = (AtA[0] * Atb[1] - AtA[2] * Atb[0]) / det;
    }
  } else if (dim == 3) {
    // Use Cramer's rule for 3x3 system
    double a11 = AtA[0], a12 = AtA[1], a13 = AtA[2];
    double a21 = AtA[3], a22 = AtA[4], a23 = AtA[5];
    double a31 = AtA[6], a32 = AtA[7], a33 = AtA[8];

    double det = a11 * (a22 * a33 - a23 * a32)
               - a12 * (a21 * a33 - a23 * a31)
               + a13 * (a21 * a32 - a22 * a31);

    if (std::abs(det) > 1e-20) {
      // Compute inverse times Atb
      gradient[0] = ((a22 * a33 - a23 * a32) * Atb[0]
                   - (a12 * a33 - a13 * a32) * Atb[1]
                   + (a12 * a23 - a13 * a22) * Atb[2]) / det;
      gradient[1] = (-(a21 * a33 - a23 * a31) * Atb[0]
                   + (a11 * a33 - a13 * a31) * Atb[1]
                   - (a11 * a23 - a13 * a21) * Atb[2]) / det;
      gradient[2] = ((a21 * a32 - a22 * a31) * Atb[0]
                   - (a11 * a32 - a12 * a31) * Atb[1]
                   + (a11 * a22 - a12 * a21) * Atb[2]) / det;
    }
  }

  return gradient;
}

/**
 * @brief Compute cell gradient using simple L2 projection (unweighted averaging)
 *
 * Simpler alternative to LSQ that uses equal-weighted averaging of neighbor
 * gradient estimates. Less accurate than LSQ but faster and matches MFEM's L2ZZ.
 *
 * @param mesh The mesh
 * @param cell_id Cell index
 * @param field_values Cell-centered field values
 * @return Gradient vector (size = mesh.dim())
 */
inline std::vector<double> compute_cell_gradient_l2(
    const MeshBase& mesh,
    index_t cell_id,
    const std::vector<double>& field_values) {

  size_t dim = mesh.dim();
  std::vector<double> gradient(dim, 0.0);

  auto center = mesh.cell_center(cell_id);
  double center_value = field_values[cell_id];

  auto neighbors = mesh.cell_neighbors(cell_id);
  if (neighbors.empty()) {
    return gradient;  // Isolated cell, zero gradient
  }

  // Simple unweighted averaging (L2 projection)
  int count = 0;
  for (index_t neighbor_id : neighbors) {
    if (neighbor_id < 0 || neighbor_id >= static_cast<index_t>(mesh.n_cells())) {
      continue;
    }

    auto neighbor_center = mesh.cell_center(neighbor_id);
    double neighbor_value = field_values[neighbor_id];

    std::vector<double> dir(dim);
    double dist = 0.0;
    for (size_t d = 0; d < dim; ++d) {
      dir[d] = neighbor_center[d] - center[d];
      dist += dir[d] * dir[d];
    }
    dist = std::sqrt(dist);

    if (dist < 1e-20) continue;

    // Unweighted directional derivative
    double deriv = (neighbor_value - center_value) / dist;
    for (size_t d = 0; d < dim; ++d) {
      gradient[d] += deriv * (dir[d] / dist);
    }
    count++;
  }

  // Average over neighbors
  if (count > 0) {
    for (size_t d = 0; d < dim; ++d) {
      gradient[d] /= count;
    }
  }

  return gradient;
}

/**
 * @brief Recover gradients at vertices using Superconvergent Patch Recovery (SPR)
 *
 * For each vertex, computes a volume-weighted average of gradients from
 * incident cells. This produces a smooth gradient field that is
 * superconvergent for certain element types and integration points.
 *
 * @param mesh The mesh
 * @param cell_gradients Cell-centered gradients (size = n_cells x dim)
 * @return Vertex gradients (size = n_vertices x dim)
 */
inline std::vector<std::vector<double>> recover_vertex_gradients_spr(
    const MeshBase& mesh,
    const std::vector<std::vector<double>>& cell_gradients) {

  size_t num_vertices = mesh.n_vertices();
  size_t dim = mesh.dim();

  std::vector<std::vector<double>> vertex_gradients(num_vertices, std::vector<double>(dim, 0.0));
  std::vector<double> vertex_weights(num_vertices, 0.0);

  // For each vertex, average gradients from incident cells
  for (size_t v = 0; v < num_vertices; ++v) {
    auto incident_cells = mesh.vertex_cells(static_cast<index_t>(v));

    if (incident_cells.empty()) continue;

    for (index_t cell_id : incident_cells) {
      if (cell_id < 0 || cell_id >= static_cast<index_t>(mesh.n_cells())) {
        continue;
      }

      // Weight by cell volume
      double weight = mesh.cell_measure(cell_id);

      for (size_t d = 0; d < dim; ++d) {
        vertex_gradients[v][d] += weight * cell_gradients[cell_id][d];
      }
      vertex_weights[v] += weight;
    }

    // Normalize by total weight
    if (vertex_weights[v] > 1e-20) {
      for (size_t d = 0; d < dim; ++d) {
        vertex_gradients[v][d] /= vertex_weights[v];
      }
    }
  }

  return vertex_gradients;
}

/**
 * @brief Sanitize indicators by removing NaN/Inf values
 *
 * Replaces non-finite values (NaN, Inf, -Inf) with zero.
 * This prevents error propagation in downstream marking/refinement steps.
 *
 * @param indicators Cell-wise error indicators (modified in-place)
 */
inline void sanitize_indicators(std::vector<double>& indicators) {
  for (size_t i = 0; i < indicators.size(); ++i) {
    if (!std::isfinite(indicators[i])) {
      indicators[i] = 0.0;  // Clamp NaN/Inf to zero
    }
  }
}

} // namespace detail
} // namespace svmp

#endif // SVMP_ERROR_ESTIMATOR_DETAIL_H
