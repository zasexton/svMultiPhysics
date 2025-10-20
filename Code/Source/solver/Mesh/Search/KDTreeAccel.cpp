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

#include "KDTreeAccel.h"
#include "SearchBuilders.h"
#include "../Core/MeshBase.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace svmp {

// ---- Building ----

void KDTreeAccel::build(const MeshBase& mesh,
                        Configuration cfg,
                        const MeshSearch::SearchConfig& config) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Clear existing structure
  clear();

  // Store configuration
  built_cfg_ = cfg;
  max_depth_ = config.max_depth;
  min_points_per_leaf_ = config.min_entities_per_leaf;

  // Extract vertex coordinates
  vertices_ = search::SearchBuilders::extract_vertex_coords(mesh, cfg);
  size_t n_vertices = vertices_.size();

  if (n_vertices == 0) {
    is_built_ = true;
    return;
  }

  // Create index array
  vertex_indices_.resize(n_vertices);
  std::iota(vertex_indices_.begin(), vertex_indices_.end(), 0);

  // Build the tree
  root_ = build_node(vertex_indices_, 0, n_vertices, 0);

  // Build auxiliary structures for cell queries
  cell_aabbs_ = search::SearchBuilders::compute_all_cell_aabbs(mesh, cfg);
  build_vertex_to_cells(mesh);

  is_built_ = true;

  // Update statistics
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  stats_.build_time_ms = duration.count();
  stats_.n_entities = n_vertices + mesh.n_cells();
  stats_.tree_depth = compute_tree_depth(root_.get());
  stats_.memory_bytes = sizeof(KDNode) * count_nodes(root_.get()) +
                       sizeof(std::array<real_t,3>) * vertices_.size() +
                       sizeof(search::AABB) * cell_aabbs_.size();
}

void KDTreeAccel::clear() {
  root_.reset();
  vertices_.clear();
  vertex_indices_.clear();
  cell_aabbs_.clear();
  vertex_to_cells_.clear();
  is_built_ = false;
  stats_ = SearchStats();
}

// ---- Tree construction ----

std::unique_ptr<KDTreeAccel::KDNode> KDTreeAccel::build_node(
    std::vector<index_t>& indices,
    int start, int end,
    int depth) {

  auto node = std::make_unique<KDNode>();

  // Compute bounds for this node
  node->bounds = compute_bounds(indices, start, end);

  int n_points = end - start;

  // Check if we should create a leaf node
  if (n_points <= min_points_per_leaf_ || depth >= max_depth_) {
    node->is_leaf = true;
    node->point_indices.assign(indices.begin() + start, indices.begin() + end);
    return node;
  }

  // Choose split axis and partition
  int axis = choose_split_axis(indices, start, end, depth);
  node->split_axis = axis;

  // Partition points around median
  int mid = partition_points(indices, start, end, axis, node->split_value);

  // Recursively build children
  if (mid > start && mid < end) {
    node->left = build_node(indices, start, mid, depth + 1);
    node->right = build_node(indices, mid, end, depth + 1);
  } else {
    // Degenerate case - all points have same coordinate on split axis
    // Make this a leaf node
    node->is_leaf = true;
    node->point_indices.assign(indices.begin() + start, indices.begin() + end);
  }

  return node;
}

int KDTreeAccel::choose_split_axis(const std::vector<index_t>& indices,
                                   int start, int end,
                                   int depth) {
  // Use spread-based heuristic: choose axis with maximum spread
  std::array<real_t, 3> min_coords = {1e30, 1e30, 1e30};
  std::array<real_t, 3> max_coords = {-1e30, -1e30, -1e30};

  for (int i = start; i < end; ++i) {
    const auto& v = vertices_[indices[i]];
    for (int d = 0; d < 3; ++d) {
      min_coords[d] = std::min(min_coords[d], v[d]);
      max_coords[d] = std::max(max_coords[d], v[d]);
    }
  }

  // Find axis with maximum spread
  int best_axis = 0;
  real_t max_spread = 0;
  for (int d = 0; d < 3; ++d) {
    real_t spread = max_coords[d] - min_coords[d];
    if (spread > max_spread) {
      max_spread = spread;
      best_axis = d;
    }
  }

  return best_axis;
}

int KDTreeAccel::partition_points(std::vector<index_t>& indices,
                                  int start, int end,
                                  int axis,
                                  real_t& split_value) {
  int n = end - start;
  int mid = start + n / 2;

  // Use nth_element to partition around median
  std::nth_element(indices.begin() + start,
                  indices.begin() + mid,
                  indices.begin() + end,
                  [this, axis](index_t a, index_t b) {
                    return vertices_[a][axis] < vertices_[b][axis];
                  });

  // Set split value to median coordinate
  split_value = vertices_[indices[mid]][axis];

  // Handle duplicates - ensure left child gets all points <= split_value
  int split_pos = mid;
  while (split_pos < end && vertices_[indices[split_pos]][axis] <= split_value) {
    ++split_pos;
  }

  return split_pos;
}

search::AABB KDTreeAccel::compute_bounds(const std::vector<index_t>& indices,
                                        int start, int end) const {
  search::AABB bounds;
  for (int i = start; i < end; ++i) {
    bounds.include(vertices_[indices[i]]);
  }
  return bounds;
}

// ---- Point location ----

PointLocateResult KDTreeAccel::locate_point(const MeshBase& mesh,
                                           const std::array<real_t,3>& point,
                                           index_t hint_cell) const {
  if (!is_built_) {
    return locate_point_linear(mesh, point);
  }

  stats_.query_count++;

  // Try hint cell first
  if (hint_cell >= 0 && hint_cell < static_cast<index_t>(cell_aabbs_.size())) {
    if (cell_aabbs_[hint_cell].contains(point)) {
      auto vertex_coords = search::SearchBuilders::get_cell_vertex_coords(
          mesh, hint_cell, built_cfg_);
      auto shape = mesh.cell_shape(hint_cell);

      if (search::point_in_cell(point, shape, vertex_coords)) {
        PointLocateResult result;
        result.found = true;
        result.cell_id = hint_cell;
        result.xi = search::SearchBuilders::compute_parametric_coords(
            mesh, hint_cell, point, built_cfg_);
        stats_.hit_count++;
        return result;
      }
    }
  }

  // Find nearest vertices and check their cells
  const size_t k_neighbors = 8;
  auto nearest = k_nearest_vertices(mesh, point, k_neighbors);

  std::unordered_set<index_t> checked_cells;

  for (const auto& [vertex_idx, dist] : nearest) {
    // Get cells containing this vertex
    if (vertex_idx < vertex_to_cells_.size()) {
      for (index_t cell_id : vertex_to_cells_[vertex_idx]) {
        if (checked_cells.count(cell_id) > 0) continue;
        checked_cells.insert(cell_id);

        if (cell_aabbs_[cell_id].contains(point)) {
          auto vertex_coords = search::SearchBuilders::get_cell_vertex_coords(
              mesh, cell_id, built_cfg_);
          auto shape = mesh.cell_shape(cell_id);

          if (search::point_in_cell(point, shape, vertex_coords)) {
            PointLocateResult result;
            result.found = true;
            result.cell_id = cell_id;
            result.xi = search::SearchBuilders::compute_parametric_coords(
                mesh, cell_id, point, built_cfg_);
            stats_.hit_count++;
            return result;
          }
        }
      }
    }
  }

  // Not found
  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;
  return result;
}

std::vector<PointLocateResult> KDTreeAccel::locate_points(
    const MeshBase& mesh,
    const std::vector<std::array<real_t,3>>& points) const {

  std::vector<PointLocateResult> results;
  results.reserve(points.size());

  index_t last_cell = -1;
  for (const auto& point : points) {
    auto result = locate_point(mesh, point, last_cell);
    if (result.found) {
      last_cell = result.cell_id;
    }
    results.push_back(result);
  }

  return results;
}

// ---- Nearest neighbor queries ----

std::pair<index_t, real_t> KDTreeAccel::nearest_vertex(
    const MeshBase& mesh,
    const std::array<real_t,3>& point) const {

  if (!is_built_ || vertices_.empty()) {
    return {INVALID_INDEX, std::numeric_limits<real_t>::max()};
  }

  stats_.query_count++;

  index_t best_idx = INVALID_INDEX;
  real_t best_dist_sq = std::numeric_limits<real_t>::max();

  nearest_neighbor_recursive(root_.get(), point, best_idx, best_dist_sq);

  if (best_idx != INVALID_INDEX) {
    stats_.hit_count++;
  }

  return {best_idx, std::sqrt(best_dist_sq)};
}

void KDTreeAccel::nearest_neighbor_recursive(
    const KDNode* node,
    const std::array<real_t,3>& point,
    index_t& best_idx,
    real_t& best_dist_sq) const {

  if (!node) return;

  if (node->is_leaf) {
    // Check all points in leaf
    for (index_t idx : node->point_indices) {
      const auto& v = vertices_[idx];
      real_t dist_sq = 0;
      for (int d = 0; d < 3; ++d) {
        real_t dx = v[d] - point[d];
        dist_sq += dx * dx;
      }

      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_idx = idx;
      }
    }
    return;
  }

  // Check which side of split to search first
  bool search_left_first = point[node->split_axis] <= node->split_value;
  const KDNode* first_child = search_left_first ? node->left.get() : node->right.get();
  const KDNode* second_child = search_left_first ? node->right.get() : node->left.get();

  // Search closer child first
  nearest_neighbor_recursive(first_child, point, best_idx, best_dist_sq);

  // Check if we need to search the other child
  real_t dist_to_split = point[node->split_axis] - node->split_value;
  if (dist_to_split * dist_to_split < best_dist_sq) {
    nearest_neighbor_recursive(second_child, point, best_idx, best_dist_sq);
  }
}

std::vector<std::pair<index_t, real_t>> KDTreeAccel::k_nearest_vertices(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    size_t k) const {

  if (!is_built_ || vertices_.empty()) {
    return {};
  }

  stats_.query_count++;

  // Max heap to keep k smallest distances
  std::priority_queue<std::pair<real_t, index_t>> max_heap;

  k_nearest_recursive(root_.get(), point, k, max_heap);

  // Extract results
  std::vector<std::pair<index_t, real_t>> results;
  while (!max_heap.empty()) {
    auto [dist_sq, idx] = max_heap.top();
    max_heap.pop();
    results.push_back({idx, std::sqrt(dist_sq)});
  }

  std::reverse(results.begin(), results.end());

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

void KDTreeAccel::k_nearest_recursive(
    const KDNode* node,
    const std::array<real_t,3>& point,
    size_t k,
    std::priority_queue<std::pair<real_t, index_t>>& max_heap) const {

  if (!node) return;

  if (node->is_leaf) {
    // Check all points in leaf
    for (index_t idx : node->point_indices) {
      const auto& v = vertices_[idx];
      real_t dist_sq = 0;
      for (int d = 0; d < 3; ++d) {
        real_t dx = v[d] - point[d];
        dist_sq += dx * dx;
      }

      if (max_heap.size() < k) {
        max_heap.push({dist_sq, idx});
      } else if (dist_sq < max_heap.top().first) {
        max_heap.pop();
        max_heap.push({dist_sq, idx});
      }
    }
    return;
  }

  // Check which side of split to search first
  bool search_left_first = point[node->split_axis] <= node->split_value;
  const KDNode* first_child = search_left_first ? node->left.get() : node->right.get();
  const KDNode* second_child = search_left_first ? node->right.get() : node->left.get();

  // Search closer child first
  k_nearest_recursive(first_child, point, k, max_heap);

  // Check if we need to search the other child
  real_t dist_to_split = point[node->split_axis] - node->split_value;
  real_t dist_to_split_sq = dist_to_split * dist_to_split;

  if (max_heap.size() < k || dist_to_split_sq < max_heap.top().first) {
    k_nearest_recursive(second_child, point, k, max_heap);
  }
}

std::vector<index_t> KDTreeAccel::vertices_in_radius(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    real_t radius) const {

  if (!is_built_ || vertices_.empty()) {
    return {};
  }

  stats_.query_count++;

  std::vector<index_t> results;
  real_t radius_sq = radius * radius;

  radius_search_recursive(root_.get(), point, radius_sq, results);

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

void KDTreeAccel::radius_search_recursive(
    const KDNode* node,
    const std::array<real_t,3>& point,
    real_t radius_sq,
    std::vector<index_t>& results) const {

  if (!node) return;

  // Early rejection if node bounds don't intersect sphere
  real_t dist_to_bounds = distance_to_node_bounds(node, point);
  if (dist_to_bounds * dist_to_bounds > radius_sq) {
    return;
  }

  if (node->is_leaf) {
    // Check all points in leaf
    for (index_t idx : node->point_indices) {
      const auto& v = vertices_[idx];
      real_t dist_sq = 0;
      for (int d = 0; d < 3; ++d) {
        real_t dx = v[d] - point[d];
        dist_sq += dx * dx;
      }

      if (dist_sq <= radius_sq) {
        results.push_back(idx);
      }
    }
    return;
  }

  // Search both children
  radius_search_recursive(node->left.get(), point, radius_sq, results);
  radius_search_recursive(node->right.get(), point, radius_sq, results);
}

// ---- Ray intersection ----

RayIntersectResult KDTreeAccel::intersect_ray(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  // KD-tree is primarily for vertex queries
  // For ray intersection, we'd need a different structure or fallback
  RayIntersectResult result;
  result.found = false;
  result.face_id = -1;
  result.t = -1.0;

  // Could implement by finding cells near ray and testing them
  // For now, return not implemented

  return result;
}

std::vector<RayIntersectResult> KDTreeAccel::intersect_ray_all(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  // Not efficiently supported by KD-tree
  return {};
}

// ---- Region queries ----

std::vector<index_t> KDTreeAccel::cells_in_box(
    const MeshBase& mesh,
    const std::array<real_t,3>& box_min,
    const std::array<real_t,3>& box_max) const {

  if (!is_built_) {
    return {};
  }

  stats_.query_count++;

  // Find vertices in box
  search::AABB box(box_min, box_max);
  std::vector<index_t> vertices_in_box;
  box_search_recursive(root_.get(), box, vertices_in_box);

  // Get unique cells from vertices
  std::unordered_set<index_t> unique_cells;
  for (index_t v_idx : vertices_in_box) {
    if (v_idx < vertex_to_cells_.size()) {
      for (index_t cell_id : vertex_to_cells_[v_idx]) {
        // Check if cell AABB overlaps box
        if (cell_aabbs_[cell_id].overlaps(box)) {
          unique_cells.insert(cell_id);
        }
      }
    }
  }

  std::vector<index_t> results(unique_cells.begin(), unique_cells.end());

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

void KDTreeAccel::box_search_recursive(
    const KDNode* node,
    const search::AABB& box,
    std::vector<index_t>& results) const {

  if (!node) return;

  // Early rejection if node bounds don't overlap box
  if (!node->bounds.overlaps(box)) {
    return;
  }

  if (node->is_leaf) {
    // Check all points in leaf
    for (index_t idx : node->point_indices) {
      if (box.contains(vertices_[idx])) {
        results.push_back(idx);
      }
    }
    return;
  }

  // Check if we need to search children
  if (box.min[node->split_axis] <= node->split_value) {
    box_search_recursive(node->left.get(), box, results);
  }
  if (box.max[node->split_axis] > node->split_value) {
    box_search_recursive(node->right.get(), box, results);
  }
}

std::vector<index_t> KDTreeAccel::cells_in_sphere(
    const MeshBase& mesh,
    const std::array<real_t,3>& center,
    real_t radius) const {

  if (!is_built_) {
    return {};
  }

  stats_.query_count++;

  // Find vertices in sphere
  auto vertices_in_sphere = vertices_in_radius(mesh, center, radius);

  // Get unique cells from vertices
  std::unordered_set<index_t> unique_cells;
  for (index_t v_idx : vertices_in_sphere) {
    if (v_idx < vertex_to_cells_.size()) {
      for (index_t cell_id : vertex_to_cells_[v_idx]) {
        // Check if cell AABB intersects sphere
        if (search::aabb_sphere_overlap(cell_aabbs_[cell_id], center, radius)) {
          unique_cells.insert(cell_id);
        }
      }
    }
  }

  std::vector<index_t> results(unique_cells.begin(), unique_cells.end());

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

// ---- Helper methods ----

real_t KDTreeAccel::distance_to_node_bounds(const KDNode* node,
                                           const std::array<real_t,3>& point) const {
  if (!node) return std::numeric_limits<real_t>::max();
  return search::point_aabb_distance(point, node->bounds);
}

void KDTreeAccel::build_vertex_to_cells(const MeshBase& mesh) {
  size_t n_vertices = vertices_.size();
  vertex_to_cells_.resize(n_vertices);

  size_t n_cells = mesh.n_cells();
  for (size_t c = 0; c < n_cells; ++c) {
    auto cell_vertices = mesh.cell_vertices(c);
    for (index_t v : cell_vertices) {
      if (v < n_vertices) {
        vertex_to_cells_[v].push_back(c);
      }
    }
  }
}

std::vector<index_t> KDTreeAccel::get_cells_from_vertices(
    const std::vector<index_t>& vertex_indices) const {

  std::unordered_set<index_t> unique_cells;

  for (index_t v_idx : vertex_indices) {
    if (v_idx < vertex_to_cells_.size()) {
      for (index_t cell_id : vertex_to_cells_[v_idx]) {
        unique_cells.insert(cell_id);
      }
    }
  }

  return std::vector<index_t>(unique_cells.begin(), unique_cells.end());
}

PointLocateResult KDTreeAccel::locate_point_linear(
    const MeshBase& mesh,
    const std::array<real_t,3>& point) const {

  // Fallback linear search
  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;

  size_t n_cells = mesh.n_cells();
  for (size_t c = 0; c < n_cells; ++c) {
    auto vertex_coords = search::SearchBuilders::get_cell_vertex_coords(
        mesh, c, built_cfg_);
    auto shape = mesh.cell_shape(c);

    if (search::point_in_cell(point, shape, vertex_coords)) {
      result.found = true;
      result.cell_id = c;
      result.xi = search::SearchBuilders::compute_parametric_coords(
          mesh, c, point, built_cfg_);
      return result;
    }
  }

  return result;
}

real_t KDTreeAccel::compute_balance_factor(const KDNode* node) const {
  if (!node || node->is_leaf) return 1.0;

  size_t left_count = count_nodes(node->left.get());
  size_t right_count = count_nodes(node->right.get());

  if (left_count == 0 || right_count == 0) return 0.0;

  return static_cast<real_t>(std::min(left_count, right_count)) /
         std::max(left_count, right_count);
}

size_t KDTreeAccel::count_nodes(const KDNode* node) const {
  if (!node) return 0;
  if (node->is_leaf) return 1;
  return 1 + count_nodes(node->left.get()) + count_nodes(node->right.get());
}

int KDTreeAccel::compute_tree_depth(const KDNode* node) const {
  if (!node || node->is_leaf) return 1;
  return 1 + std::max(compute_tree_depth(node->left.get()),
                     compute_tree_depth(node->right.get()));
}

} // namespace svmp