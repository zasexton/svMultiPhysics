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

#include "MeshSearch.h"
#include "../Core/MeshBase.h"
#include "../Geometry/MeshGeometry.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <cmath>
#include <unordered_set>

namespace svmp {

// ---- Point location ----

PointLocateResult MeshSearch::locate_point(const MeshBase& mesh,
                                          const std::array<real_t,3>& point,
                                          Configuration cfg,
                                          index_t hint_cell) {
  // Simple linear search implementation
  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;

  size_t n_cells = mesh.n_cells();

  // Try hint cell first if provided
  if (hint_cell >= 0 && hint_cell < static_cast<index_t>(n_cells)) {
    if (point_in_cell(mesh, point, hint_cell, cfg)) {
      result.cell_id = hint_cell;
      result.found = true;
      result.xi = compute_parametric_coords(mesh, hint_cell, point, cfg);
      return result;
    }
  }

  // Linear search through all cells
  for (size_t c = 0; c < n_cells; ++c) {
    if (point_in_cell(mesh, point, static_cast<index_t>(c), cfg)) {
      result.cell_id = static_cast<index_t>(c);
      result.found = true;
      result.xi = compute_parametric_coords(mesh, result.cell_id, point, cfg);
      return result;
    }
  }

  return result;
}

std::vector<PointLocateResult> MeshSearch::locate_points(const MeshBase& mesh,
                                                        const std::vector<std::array<real_t,3>>& points,
                                                        Configuration cfg) {
  std::vector<PointLocateResult> results;
  results.reserve(points.size());

  for (const auto& point : points) {
    results.push_back(locate_point(mesh, point, cfg));
  }

  return results;
}

bool MeshSearch::contains_point(const MeshBase& mesh,
                               const std::array<real_t,3>& point,
                               Configuration cfg) {
  return locate_point(mesh, point, cfg).found;
}

// ---- Nearest neighbor search ----

std::pair<index_t, real_t> MeshSearch::nearest_vertex(const MeshBase& mesh,
                                                     const std::array<real_t,3>& point,
                                                     Configuration cfg) {
  index_t nearest_idx = INVALID_INDEX;
  real_t min_dist = std::numeric_limits<real_t>::max();

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  for (size_t n = 0; n < n_vertices; ++n) {
    real_t dist_sq = 0;
    for (int d = 0; d < dim; ++d) {
      real_t dx = coords[n * dim + d] - point[d];
      dist_sq += dx * dx;
    }

    if (dist_sq < min_dist * min_dist) {
      nearest_idx = static_cast<index_t>(n);
      min_dist = std::sqrt(dist_sq);
    }
  }

  return {nearest_idx, min_dist};
}

std::vector<std::pair<index_t, real_t>> MeshSearch::k_nearest_vertices(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    size_t k,
    Configuration cfg) {
  using HeapEntry = std::pair<real_t, index_t>;
  std::priority_queue<HeapEntry> max_heap;

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  for (size_t n = 0; n < n_vertices; ++n) {
    real_t dist_sq = 0;
    for (int d = 0; d < dim; ++d) {
      real_t dx = coords[n * dim + d] - point[d];
      dist_sq += dx * dx;
    }
    real_t dist = std::sqrt(dist_sq);

    if (max_heap.size() < k) {
      max_heap.push({dist, static_cast<index_t>(n)});
    } else if (dist < max_heap.top().first) {
      max_heap.pop();
      max_heap.push({dist, static_cast<index_t>(n)});
    }
  }

  // Extract results
  std::vector<std::pair<index_t, real_t>> results;
  while (!max_heap.empty()) {
    auto [dist, idx] = max_heap.top();
    max_heap.pop();
    results.push_back({idx, dist});
  }

  std::reverse(results.begin(), results.end());
  return results;
}

std::vector<index_t> MeshSearch::vertices_in_radius(const MeshBase& mesh,
                                                   const std::array<real_t,3>& point,
                                                   real_t radius,
                                                   Configuration cfg) {
  std::vector<index_t> vertices;
  real_t radius_sq = radius * radius;

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  for (size_t n = 0; n < n_vertices; ++n) {
    real_t dist_sq = 0;
    for (int d = 0; d < dim; ++d) {
      real_t dx = coords[n * dim + d] - point[d];
      dist_sq += dx * dx;
    }

    if (dist_sq <= radius_sq) {
      vertices.push_back(static_cast<index_t>(n));
    }
  }

  return vertices;
}

std::pair<index_t, real_t> MeshSearch::nearest_cell(const MeshBase& mesh,
                                                   const std::array<real_t,3>& point,
                                                   Configuration cfg) {
  index_t nearest_idx = INVALID_INDEX;
  real_t min_dist = std::numeric_limits<real_t>::max();

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    real_t dist = distance_to_cell(mesh, point, static_cast<index_t>(c), cfg);
    if (dist < min_dist) {
      nearest_idx = static_cast<index_t>(c);
      min_dist = dist;
    }
  }

  return {nearest_idx, min_dist};
}

// ---- Ray intersection ----

RayIntersectResult MeshSearch::intersect_ray(const MeshBase& mesh,
                                            const std::array<real_t,3>& origin,
                                            const std::array<real_t,3>& direction,
                                            Configuration cfg,
                                            real_t max_distance) {
  RayIntersectResult result;
  result.found = false;
  result.face_id = -1;
  result.t = -1.0;

  // TODO: Implement ray-mesh intersection
  // This requires testing each face for ray-triangle intersection

  return result;
}

std::vector<RayIntersectResult> MeshSearch::intersect_ray_all(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    Configuration cfg,
    real_t max_distance) {
  std::vector<RayIntersectResult> results;

  // TODO: Implement finding all ray-mesh intersections

  return results;
}

// ---- Distance queries ----

real_t MeshSearch::signed_distance(const MeshBase& mesh,
                                  const std::array<real_t,3>& point,
                                  Configuration cfg) {
  // Simple unsigned distance for now
  // TODO: Implement proper signed distance computation
  auto [face_id, dist] = closest_boundary_point(mesh, point, cfg);
  return dist;
}

std::pair<std::array<real_t,3>, index_t> MeshSearch::closest_boundary_point(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    Configuration cfg) {
  std::array<real_t,3> closest_pt = {0, 0, 0};
  index_t closest_face = INVALID_INDEX;
  real_t min_dist = std::numeric_limits<real_t>::max();

  size_t n_faces = mesh.n_faces();

  for (size_t f = 0; f < n_faces; ++f) {
    auto face_cells = mesh.face_cells(static_cast<index_t>(f));

    // Only check boundary faces
    if (face_cells[1] == INVALID_INDEX) {
      auto center = mesh.face_center(static_cast<index_t>(f), cfg);

      real_t dist_sq = 0;
      for (int d = 0; d < 3; ++d) {
        real_t dx = center[d] - point[d];
        dist_sq += dx * dx;
      }

      if (dist_sq < min_dist * min_dist) {
        min_dist = std::sqrt(dist_sq);
        closest_face = static_cast<index_t>(f);
        closest_pt = center;
      }
    }
  }

  return {closest_pt, closest_face};
}

// ---- Search structure management ----

void MeshSearch::build_search_structure(const MeshBase& mesh,
                                       const SearchConfig& config,
                                       Configuration cfg) {
  // Delegate to MeshBase holder (placeholder in current implementation)
  mesh.build_search_structure(cfg);
}

void MeshSearch::clear_search_structure(const MeshBase& mesh) {
  mesh.clear_search_structure();
}

bool MeshSearch::has_search_structure(const MeshBase& mesh) {
  return mesh.has_search_structure();
}

// ---- Spatial queries ----

std::vector<index_t> MeshSearch::cells_in_box(const MeshBase& mesh,
                                             const std::array<real_t,3>& box_min,
                                             const std::array<real_t,3>& box_max,
                                             Configuration cfg) {
  std::vector<index_t> cells;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto center = mesh.cell_center(static_cast<index_t>(c), cfg);

    bool inside = true;
    for (int d = 0; d < 3; ++d) {
      if (center[d] < box_min[d] || center[d] > box_max[d]) {
        inside = false;
        break;
      }
    }

    if (inside) {
      cells.push_back(static_cast<index_t>(c));
    }
  }

  return cells;
}

std::vector<index_t> MeshSearch::cells_in_sphere(const MeshBase& mesh,
                                                const std::array<real_t,3>& center,
                                                real_t radius,
                                                Configuration cfg) {
  std::vector<index_t> cells;
  real_t radius_sq = radius * radius;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto cell_center = mesh.cell_center(static_cast<index_t>(c), cfg);

    real_t dist_sq = 0;
    for (int d = 0; d < 3; ++d) {
      real_t dx = cell_center[d] - center[d];
      dist_sq += dx * dx;
    }

    if (dist_sq <= radius_sq) {
      cells.push_back(static_cast<index_t>(c));
    }
  }

  return cells;
}

// ---- Parametric coordinates ----

std::array<real_t,3> MeshSearch::compute_parametric_coords(const MeshBase& mesh,
                                                          index_t cell,
                                                          const std::array<real_t,3>& point,
                                                          Configuration cfg) {
  // TODO: Implement computation of parametric coordinates
  // This requires solving the inverse mapping from physical to reference space
  return {0, 0, 0};
}

bool MeshSearch::is_inside_reference_element(const CellShape& shape,
                                            const std::array<real_t,3>& xi) {
  // Check if parametric coordinates are inside reference element
  switch (shape.family) {
    case CellFamily::Line:
      return xi[0] >= -1 && xi[0] <= 1;

    case CellFamily::Triangle:
      return xi[0] >= 0 && xi[1] >= 0 && (xi[0] + xi[1]) <= 1;

    case CellFamily::Quad:
      return xi[0] >= -1 && xi[0] <= 1 &&
             xi[1] >= -1 && xi[1] <= 1;

    case CellFamily::Tetra:
      return xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 &&
             (xi[0] + xi[1] + xi[2]) <= 1;

    case CellFamily::Hex:
      return xi[0] >= -1 && xi[0] <= 1 &&
             xi[1] >= -1 && xi[1] <= 1 &&
             xi[2] >= -1 && xi[2] <= 1;

    default:
      return false;
  }
}

// ---- Walking algorithms ----

std::vector<index_t> MeshSearch::walk_to_point(const MeshBase& mesh,
                                              index_t start_cell,
                                              const std::array<real_t,3>& target_point,
                                              Configuration cfg) {
  std::vector<index_t> path;
  path.push_back(start_cell);

  // TODO: Implement walking algorithm
  // This would walk through the mesh from start_cell toward target_point

  return path;
}

// ---- Helper methods ----

bool MeshSearch::point_in_cell(const MeshBase& mesh,
                              const std::array<real_t,3>& point,
                              index_t cell,
                              Configuration cfg) {
  // Simple check: compute parametric coordinates and check if inside reference element
  auto xi = compute_parametric_coords(mesh, cell, point, cfg);
  auto shape = mesh.cell_shape(cell);
  return is_inside_reference_element(shape, xi);
}

real_t MeshSearch::distance_to_cell(const MeshBase& mesh,
                                   const std::array<real_t,3>& point,
                                   index_t cell,
                                   Configuration cfg) {
  // Simple distance to cell center
  auto center = mesh.cell_center(cell, cfg);

  real_t dist_sq = 0;
  for (int d = 0; d < 3; ++d) {
    real_t dx = center[d] - point[d];
    dist_sq += dx * dx;
  }

  return std::sqrt(dist_sq);
}

} // namespace svmp
