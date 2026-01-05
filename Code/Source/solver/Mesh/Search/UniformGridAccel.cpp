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

#include "UniformGridAccel.h"
#include "SearchBuilders.h"
#include "../Core/MeshBase.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_set>
#include <queue>

namespace svmp {

// ---- Building ----

void UniformGridAccel::build(const MeshBase& mesh,
                             Configuration cfg,
                             const MeshSearch::SearchConfig& config) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Clear existing structure
  clear();

  // Store configuration
  built_cfg_ = cfg;

  // Determine grid resolution
  int resolution = config.grid_resolution;
  if (resolution <= 0) {
    resolution = search::SearchBuilders::estimate_grid_resolution(mesh, cfg);
  }

  // Build structures
  build_cell_grid(mesh, cfg, resolution);
  build_vertex_grid(mesh, cfg, resolution);
  build_boundary_triangles(mesh, cfg);

  is_built_ = true;

  // Update statistics
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  stats_.build_time_ms = duration.count();
  stats_.n_entities = mesh.n_cells() + mesh.n_vertices();
  stats_.memory_bytes = sizeof(GridCell) * grid_cells_.size() +
                       sizeof(search::AABB) * cell_aabbs_.size() +
                       sizeof(std::array<real_t,3>) * vertex_coords_.size() +
                       sizeof(BoundaryTriangle) * boundary_triangles_.size();
}

void UniformGridAccel::clear() {
  grid_cells_.clear();
  cell_aabbs_.clear();
  vertex_coords_.clear();
  boundary_triangles_.clear();
  is_built_ = false;
  stats_ = SearchStats();
}

// ---- Point location ----

PointLocateResult UniformGridAccel::locate_point(const MeshBase& mesh,
                                                const std::array<real_t,3>& point,
                                                index_t hint_cell) const {
  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;

  if (!is_built_) {
    return result;
  }

  stats_.query_count++;

  // Try hint cell first
  if (hint_cell >= 0 && hint_cell < static_cast<index_t>(cell_aabbs_.size())) {
    if (point_in_cell_cached(point, hint_cell, mesh)) {
      result.cell_id = hint_cell;
      result.found = true;
      result.xi = search::SearchBuilders::compute_parametric_coords(
          mesh, hint_cell, point, built_cfg_);
      stats_.hit_count++;
      return result;
    }
  }

  // Get candidate cells from grid
  auto candidates = get_candidate_cells(point);

  // Test each candidate
  for (index_t cell_id : candidates) {
    if (point_in_cell_cached(point, cell_id, mesh)) {
      result.cell_id = cell_id;
      result.found = true;
      result.xi = search::SearchBuilders::compute_parametric_coords(
          mesh, cell_id, point, built_cfg_);
      stats_.hit_count++;
      return result;
    }
  }

  return result;
}

std::vector<PointLocateResult> UniformGridAccel::locate_points(
    const MeshBase& mesh,
    const std::vector<std::array<real_t,3>>& points) const {

  std::vector<PointLocateResult> results;
  results.reserve(points.size());

  index_t last_cell = -1;
  for (const auto& point : points) {
    // Use last found cell as hint for next point (useful for nearby points)
    auto result = locate_point(mesh, point, last_cell);
    if (result.found) {
      last_cell = result.cell_id;
    }
    results.push_back(result);
  }

  return results;
}

// ---- Nearest neighbor ----

std::pair<index_t, real_t> UniformGridAccel::nearest_vertex(
    const MeshBase& mesh,
    const std::array<real_t,3>& point) const {

  if (!is_built_ || vertex_coords_.empty()) {
    return {INVALID_INDEX, std::numeric_limits<real_t>::max()};
  }

  stats_.query_count++;

  index_t nearest_idx = INVALID_INDEX;
  real_t min_dist = std::numeric_limits<real_t>::max();

  // Start with small radius and expand
  real_t search_radius = cell_size_[0];  // Start with one grid cell
  const auto ext = bounds_.extents();
  const real_t max_radius =
      std::sqrt(ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]);

  while (nearest_idx == INVALID_INDEX && search_radius < max_radius) {
    auto candidates = get_nearby_vertices(point, search_radius);

    for (index_t v_idx : candidates) {
      if (v_idx < vertex_coords_.size()) {
        auto& v_coord = vertex_coords_[v_idx];
        real_t dist = std::sqrt(search::dot3(search::sub3(v_coord, point),
                                            search::sub3(v_coord, point)));
        if (dist < min_dist) {
          min_dist = dist;
          nearest_idx = v_idx;
        }
      }
    }

    if (nearest_idx == INVALID_INDEX) {
      search_radius *= 2;  // Expand search radius
    }
  }

  if (nearest_idx != INVALID_INDEX) {
    stats_.hit_count++;
  }

  return {nearest_idx, min_dist};
}

std::vector<std::pair<index_t, real_t>> UniformGridAccel::k_nearest_vertices(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    size_t k) const {

  using HeapEntry = std::pair<real_t, index_t>;
  std::priority_queue<HeapEntry> max_heap;

  if (!is_built_ || vertex_coords_.empty()) {
    return {};
  }

  stats_.query_count++;

  // Progressively expand search radius
  real_t search_radius = cell_size_[0];
  const auto ext = bounds_.extents();
  const real_t max_radius =
      std::sqrt(ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]);

  while (max_heap.size() < k && search_radius < max_radius) {
    auto candidates = get_nearby_vertices(point, search_radius);

    for (index_t v_idx : candidates) {
      if (v_idx < vertex_coords_.size()) {
        auto& v_coord = vertex_coords_[v_idx];
        real_t dist = std::sqrt(search::dot3(search::sub3(v_coord, point),
                                            search::sub3(v_coord, point)));

        if (max_heap.size() < k) {
          max_heap.push({dist, v_idx});
        } else if (dist < max_heap.top().first) {
          max_heap.pop();
          max_heap.push({dist, v_idx});
        }
      }
    }

    search_radius *= 2;
  }

  // Extract results
  std::vector<std::pair<index_t, real_t>> results;
  while (!max_heap.empty()) {
    auto [dist, idx] = max_heap.top();
    max_heap.pop();
    results.push_back({idx, dist});
  }

  std::reverse(results.begin(), results.end());

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

std::vector<index_t> UniformGridAccel::vertices_in_radius(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    real_t radius) const {

  std::vector<index_t> vertices;

  if (!is_built_ || vertex_coords_.empty()) {
    return vertices;
  }

  stats_.query_count++;

  auto candidates = get_nearby_vertices(point, radius);
  real_t radius_sq = radius * radius;

  for (index_t v_idx : candidates) {
    if (v_idx < vertex_coords_.size()) {
      auto& v_coord = vertex_coords_[v_idx];
      auto diff = search::sub3(v_coord, point);
      real_t dist_sq = search::dot3(diff, diff);

      if (dist_sq <= radius_sq) {
        vertices.push_back(v_idx);
      }
    }
  }

  if (!vertices.empty()) {
    stats_.hit_count++;
  }

  return vertices;
}

// ---- Ray intersection ----

RayIntersectResult UniformGridAccel::intersect_ray(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  RayIntersectResult result;
  result.found = false;
  result.face_id = -1;
  result.t = -1.0;

  if (!is_built_ || boundary_triangles_.empty()) {
    return result;
  }

  stats_.query_count++;

  search::Ray ray(origin, search::normalize3(direction), 0.0, max_distance);

  // Walk through grid cells along ray
  auto traversed_cells = walk_ray(ray);

  real_t min_t = max_distance;
  index_t hit_face = INVALID_INDEX;
  std::array<real_t,3> hit_point;

  // Test triangles in traversed cells
  std::unordered_set<size_t> tested_triangles;

  for (size_t cell_idx : traversed_cells) {
    if (cell_idx >= grid_cells_.size()) continue;

    const auto& cell = grid_cells_[cell_idx];

    // Test boundary triangles in this cell
    for (index_t entity_id : cell.entities) {
      // Check if this is a triangle index
      if (entity_id >= mesh.n_cells() &&
          entity_id < mesh.n_cells() + boundary_triangles_.size()) {

        size_t tri_idx = entity_id - mesh.n_cells();

        if (tested_triangles.count(tri_idx) > 0) continue;
        tested_triangles.insert(tri_idx);

        const auto& tri = boundary_triangles_[tri_idx];
        real_t t;

        if (search::ray_triangle_intersect(ray, tri.vertices[0],
                                          tri.vertices[1],
                                          tri.vertices[2], t)) {
          if (t < min_t) {
            min_t = t;
            hit_face = tri.face_id;
            hit_point = ray.point_at(t);
          }
        }
      }
    }

    // Early termination if we found a hit closer than next cells
    if (hit_face != INVALID_INDEX) {
      // Could add early termination logic here
    }
  }

  if (hit_face != INVALID_INDEX) {
    result.found = true;
    result.hit = true;
    result.face_id = hit_face;
    result.t = min_t;
    result.distance = min_t;
    result.point = hit_point;
    result.hit_point = hit_point;
    stats_.hit_count++;
  }

  return result;
}

std::vector<RayIntersectResult> UniformGridAccel::intersect_ray_all(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  std::vector<RayIntersectResult> results;

  if (!is_built_ || boundary_triangles_.empty()) {
    return results;
  }

  stats_.query_count++;

  search::Ray ray(origin, search::normalize3(direction), 0.0, max_distance);

  // Walk through all grid cells along ray
  auto traversed_cells = walk_ray(ray);

  std::unordered_set<size_t> tested_triangles;

  for (size_t cell_idx : traversed_cells) {
    if (cell_idx >= grid_cells_.size()) continue;

    const auto& cell = grid_cells_[cell_idx];

    for (index_t entity_id : cell.entities) {
      if (entity_id >= mesh.n_cells() &&
          entity_id < mesh.n_cells() + boundary_triangles_.size()) {

        size_t tri_idx = entity_id - mesh.n_cells();

        if (tested_triangles.count(tri_idx) > 0) continue;
        tested_triangles.insert(tri_idx);

        const auto& tri = boundary_triangles_[tri_idx];
        real_t t;

        if (search::ray_triangle_intersect(ray, tri.vertices[0],
                                          tri.vertices[1],
                                          tri.vertices[2], t)) {
          RayIntersectResult hit;
          hit.found = true;
          hit.hit = true;
          hit.face_id = tri.face_id;
          hit.t = t;
          hit.distance = t;
          hit.point = ray.point_at(t);
          hit.hit_point = hit.point;
          results.push_back(hit);
        }
      }
    }
  }

  // Sort by t parameter
  std::sort(results.begin(), results.end(),
           [](const RayIntersectResult& a, const RayIntersectResult& b) {
             return a.t < b.t;
           });

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

// ---- Region queries ----

std::vector<index_t> UniformGridAccel::cells_in_box(
    const MeshBase& mesh,
    const std::array<real_t,3>& box_min,
    const std::array<real_t,3>& box_max) const {

  std::vector<index_t> cells;

  if (!is_built_) {
    return cells;
  }

  stats_.query_count++;

  search::AABB query_box(box_min, box_max);
  auto grid_cells = get_overlapping_cells(query_box);

  std::unordered_set<index_t> unique_cells;

  for (size_t cell_idx : grid_cells) {
    if (cell_idx >= grid_cells_.size()) continue;

    const auto& cell = grid_cells_[cell_idx];

    for (index_t entity_id : cell.entities) {
      // Check if this is a mesh cell (not vertex or triangle)
      if (entity_id < mesh.n_cells()) {
        // Check if cell AABB overlaps query box
        if (cell_aabbs_[entity_id].overlaps(query_box)) {
          unique_cells.insert(entity_id);
        }
      }
    }
  }

  cells.assign(unique_cells.begin(), unique_cells.end());

  if (!cells.empty()) {
    stats_.hit_count++;
  }

  return cells;
}

std::vector<index_t> UniformGridAccel::cells_in_sphere(
    const MeshBase& mesh,
    const std::array<real_t,3>& center,
    real_t radius) const {

  std::vector<index_t> cells;

  if (!is_built_) {
    return cells;
  }

  stats_.query_count++;

  // Get bounding box of sphere
  std::array<real_t,3> sphere_min = {
    center[0] - radius, center[1] - radius, center[2] - radius
  };
  std::array<real_t,3> sphere_max = {
    center[0] + radius, center[1] + radius, center[2] + radius
  };

  search::AABB sphere_box(sphere_min, sphere_max);
  auto grid_cells = get_overlapping_cells(sphere_box);

  std::unordered_set<index_t> unique_cells;

  for (size_t cell_idx : grid_cells) {
    if (cell_idx >= grid_cells_.size()) continue;

    const auto& cell = grid_cells_[cell_idx];

    for (index_t entity_id : cell.entities) {
      if (entity_id < mesh.n_cells()) {
        // Check if cell AABB intersects sphere
        if (search::aabb_sphere_overlap(cell_aabbs_[entity_id], center, radius)) {
          unique_cells.insert(entity_id);
        }
      }
    }
  }

  cells.assign(unique_cells.begin(), unique_cells.end());

  if (!cells.empty()) {
    stats_.hit_count++;
  }

  return cells;
}

// ---- Helper methods ----

std::array<int,3> UniformGridAccel::point_to_grid(
    const std::array<real_t,3>& point) const {

  std::array<int,3> indices;

  for (int d = 0; d < 3; ++d) {
    real_t t = (point[d] - bounds_.min[d]) / cell_size_[d];
    indices[d] = static_cast<int>(std::floor(t));
    indices[d] = std::max(0, std::min(resolution_[d] - 1, indices[d]));
  }

  return indices;
}

std::vector<size_t> UniformGridAccel::get_overlapping_cells(
    const search::AABB& aabb) const {

  std::vector<size_t> cells;

  auto min_grid = point_to_grid(aabb.min);
  auto max_grid = point_to_grid(aabb.max);

  for (int k = min_grid[2]; k <= max_grid[2]; ++k) {
    for (int j = min_grid[1]; j <= max_grid[1]; ++j) {
      for (int i = min_grid[0]; i <= max_grid[0]; ++i) {
        size_t idx = grid_index(i, j, k);
        if (idx != SIZE_MAX) {
          cells.push_back(idx);
        }
      }
    }
  }

  return cells;
}

std::vector<index_t> UniformGridAccel::get_candidate_cells(
    const std::array<real_t,3>& point) const {

  std::vector<index_t> candidates;

  auto grid_pos = point_to_grid(point);
  size_t idx = grid_index(grid_pos[0], grid_pos[1], grid_pos[2]);

  if (idx != SIZE_MAX && idx < grid_cells_.size()) {
    const auto& cell = grid_cells_[idx];

    for (index_t entity_id : cell.entities) {
      // Only return mesh cells (not vertices or triangles)
      if (entity_id < cell_aabbs_.size()) {
        candidates.push_back(entity_id);
      }
    }
  }

  return candidates;
}

std::vector<index_t> UniformGridAccel::get_nearby_vertices(
    const std::array<real_t,3>& point,
    real_t search_radius) const {

  std::vector<index_t> vertices;

  // Get grid cells within radius
  std::array<real_t,3> min_pt = {
    point[0] - search_radius,
    point[1] - search_radius,
    point[2] - search_radius
  };
  std::array<real_t,3> max_pt = {
    point[0] + search_radius,
    point[1] + search_radius,
    point[2] + search_radius
  };

  search::AABB search_box(min_pt, max_pt);
  auto cells = get_overlapping_cells(search_box);

  std::unordered_set<index_t> unique_vertices;

  for (size_t cell_idx : cells) {
    if (cell_idx >= grid_cells_.size()) continue;

    const auto& cell = grid_cells_[cell_idx];

    for (index_t entity_id : cell.entities) {
      // Check if this is a vertex index
      // Vertices are stored with offset after cells
      if (entity_id >= cell_aabbs_.size() &&
          entity_id < cell_aabbs_.size() + vertex_coords_.size()) {
        index_t v_idx = entity_id - cell_aabbs_.size();
        unique_vertices.insert(v_idx);
      }
    }
  }

  vertices.assign(unique_vertices.begin(), unique_vertices.end());
  return vertices;
}

void UniformGridAccel::build_cell_grid(const MeshBase& mesh,
                                       Configuration cfg,
                                       int resolution) {
  // Compute mesh bounds and cell AABBs
  bounds_ = search::SearchBuilders::compute_mesh_aabb(mesh, cfg);
  cell_aabbs_ = search::SearchBuilders::compute_all_cell_aabbs(mesh, cfg);

  // Set grid resolution
  resolution_ = {resolution, resolution, resolution};

  // Compute cell size
  auto extents = bounds_.extents();
  for (int d = 0; d < 3; ++d) {
    cell_size_[d] = extents[d] / resolution_[d];
    // Add small padding to avoid numerical issues
    cell_size_[d] *= 1.001;
  }

  // Allocate grid
  size_t n_grid_cells = resolution_[0] * resolution_[1] * resolution_[2];
  grid_cells_.resize(n_grid_cells);

  // Insert cells into grid
  for (size_t c = 0; c < cell_aabbs_.size(); ++c) {
    auto overlapping = get_overlapping_cells(cell_aabbs_[c]);

    for (size_t grid_idx : overlapping) {
      if (grid_idx < grid_cells_.size()) {
        grid_cells_[grid_idx].entities.push_back(static_cast<index_t>(c));
      }
    }
  }
}

void UniformGridAccel::build_vertex_grid(const MeshBase& mesh,
                                         Configuration cfg,
                                         int resolution) {
  // Extract vertex coordinates
  vertex_coords_ = search::SearchBuilders::extract_vertex_coords(mesh, cfg);

  // Insert vertices into grid
  // Use offset to distinguish from cells
  index_t vertex_offset = static_cast<index_t>(cell_aabbs_.size());

  for (size_t v = 0; v < vertex_coords_.size(); ++v) {
    auto grid_pos = point_to_grid(vertex_coords_[v]);
    size_t idx = grid_index(grid_pos[0], grid_pos[1], grid_pos[2]);

    if (idx != SIZE_MAX && idx < grid_cells_.size()) {
      grid_cells_[idx].entities.push_back(vertex_offset + static_cast<index_t>(v));
    }
  }
}

void UniformGridAccel::build_boundary_triangles(const MeshBase& mesh,
                                               Configuration cfg) {
  // Triangulate boundary for ray intersection
  auto triangles = search::SearchBuilders::triangulate_boundary(mesh, cfg);

  boundary_triangles_.clear();
  boundary_triangles_.reserve(triangles.size());

  // Use another offset for triangles
  index_t tri_offset = static_cast<index_t>(cell_aabbs_.size() + vertex_coords_.size());

  for (size_t t = 0; t < triangles.size(); ++t) {
    const auto& tri = triangles[t];

    BoundaryTriangle bt;
    bt.vertices = tri.vertices;
    bt.face_id = tri.face_id;
    boundary_triangles_.push_back(bt);

    // Compute triangle AABB and insert into grid
    search::AABB tri_aabb;
    for (const auto& v : tri.vertices) {
      tri_aabb.include(v);
    }

    auto overlapping = get_overlapping_cells(tri_aabb);
    for (size_t grid_idx : overlapping) {
      if (grid_idx < grid_cells_.size()) {
        // Store with mesh.n_cells() offset to distinguish from cells
        grid_cells_[grid_idx].entities.push_back(mesh.n_cells() + static_cast<index_t>(t));
      }
    }
  }
}

bool UniformGridAccel::point_in_cell_cached(const std::array<real_t,3>& point,
                                           index_t cell_id,
                                           const MeshBase& mesh) const {
  // First check AABB
  if (cell_id >= cell_aabbs_.size()) {
    return false;
  }

  if (!cell_aabbs_[cell_id].contains(point)) {
    return false;
  }

  // Do actual point-in-cell test
  auto shape = mesh.cell_shape(cell_id);
  auto vertices = search::SearchBuilders::get_cell_vertex_coords(mesh, cell_id, built_cfg_);

  return search::point_in_cell(point, shape, vertices);
}

std::vector<size_t> UniformGridAccel::walk_ray(const search::Ray& ray) const {
  std::vector<size_t> cells;

  // DDA algorithm for ray traversal through grid
  real_t t_near, t_far;
  if (!search::ray_aabb_intersect(ray, bounds_, t_near, t_far)) {
    return cells;  // Ray misses grid entirely
  }

  // Start point on grid boundary
  auto start_point = ray.point_at(std::max(t_near, ray.t_min));
  auto end_point = ray.point_at(std::min(t_far, ray.t_max));

  auto current = point_to_grid(start_point);
  auto end = point_to_grid(end_point);

  // Step direction for each axis
  std::array<int,3> step;
  std::array<real_t,3> t_max;
  std::array<real_t,3> t_delta;

  for (int d = 0; d < 3; ++d) {
    if (ray.direction[d] > 0) {
      step[d] = 1;
      t_max[d] = ((current[d] + 1) * cell_size_[d] + bounds_.min[d] - ray.origin[d]) /
                 ray.direction[d];
      t_delta[d] = cell_size_[d] / ray.direction[d];
    } else if (ray.direction[d] < 0) {
      step[d] = -1;
      t_max[d] = (current[d] * cell_size_[d] + bounds_.min[d] - ray.origin[d]) /
                 ray.direction[d];
      t_delta[d] = -cell_size_[d] / ray.direction[d];
    } else {
      step[d] = 0;
      t_max[d] = 1e30;
      t_delta[d] = 1e30;
    }
  }

  // Walk through grid
  const int max_steps = resolution_[0] + resolution_[1] + resolution_[2];
  int steps = 0;

  while (steps < max_steps) {
    size_t idx = grid_index(current[0], current[1], current[2]);
    if (idx != SIZE_MAX) {
      cells.push_back(idx);
    }

    // Check if we reached the end
    if (current[0] == end[0] && current[1] == end[1] && current[2] == end[2]) {
      break;
    }

    // Find next grid cell
    if (t_max[0] < t_max[1]) {
      if (t_max[0] < t_max[2]) {
        current[0] += step[0];
        if (current[0] < 0 || current[0] >= resolution_[0]) break;
        t_max[0] += t_delta[0];
      } else {
        current[2] += step[2];
        if (current[2] < 0 || current[2] >= resolution_[2]) break;
        t_max[2] += t_delta[2];
      }
    } else {
      if (t_max[1] < t_max[2]) {
        current[1] += step[1];
        if (current[1] < 0 || current[1] >= resolution_[1]) break;
        t_max[1] += t_delta[1];
      } else {
        current[2] += step[2];
        if (current[2] < 0 || current[2] >= resolution_[2]) break;
        t_max[2] += t_delta[2];
      }
    }

    steps++;
  }

  return cells;
}

} // namespace svmp
