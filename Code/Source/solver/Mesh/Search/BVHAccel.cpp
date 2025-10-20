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

#include "BVHAccel.h"
#include "SearchBuilders.h"
#include <algorithm>
#include <limits>
#include <stack>
#include <cmath>
#include <numeric>

namespace svmp {

// ---- Building ----

void BVHAccel::build(const MeshBase& mesh,
                     Configuration cfg,
                     const MeshSearch::SearchConfig& config) {
  clear();

  built_cfg_ = cfg;
  stats_ = SearchStats();
  auto start_time = std::chrono::steady_clock::now();

  // Extract mesh data based on configuration
  if (cfg == Configuration::Reference) {
    vertex_coords_ = SearchBuilders::extract_vertex_coords(mesh);
  } else {
    vertex_coords_ = SearchBuilders::extract_deformed_coords(mesh);
  }

  // Build vertex index list
  vertex_indices_.resize(vertex_coords_.size());
  std::iota(vertex_indices_.begin(), vertex_indices_.end(), 0);

  // Build vertex to cell mapping
  vertex_to_cells_.resize(vertex_coords_.size());
  for (index_t cell_id = 0; cell_id < mesh.n_cells(); ++cell_id) {
    auto conn = mesh.get_cell_connectivity(cell_id);
    for (index_t vid : conn) {
      vertex_to_cells_[vid].push_back(cell_id);
    }
  }

  // Prepare primitives based on query type
  std::vector<PrimitiveInfo> primitives;

  if (config.primary_use == MeshSearch::QueryType::RayIntersection) {
    // For ray intersection, use boundary triangles
    boundary_triangles_ = SearchBuilders::extract_boundary_triangles(mesh, vertex_coords_);

    primitives.reserve(boundary_triangles_.size());
    for (size_t i = 0; i < boundary_triangles_.size(); ++i) {
      const auto& tri = boundary_triangles_[i];
      search::AABB tri_bounds(tri.vertices[0], tri.vertices[0]);
      tri_bounds.expand(tri.vertices[1]);
      tri_bounds.expand(tri.vertices[2]);
      primitives.emplace_back(static_cast<index_t>(i), tri_bounds);
    }
  } else {
    // For general queries, use cells
    cell_indices_.resize(mesh.n_cells());
    std::iota(cell_indices_.begin(), cell_indices_.end(), 0);

    primitive_aabbs_ = SearchBuilders::compute_cell_aabbs(mesh, vertex_coords_);

    primitives.reserve(cell_indices_.size());
    for (size_t i = 0; i < cell_indices_.size(); ++i) {
      primitives.emplace_back(static_cast<index_t>(i), primitive_aabbs_[i]);
    }
  }

  // Build the BVH tree
  if (!primitives.empty()) {
    root_ = build_recursive(primitives, 0, static_cast<int>(primitives.size()), 0);
  }

  // Compute statistics
  if (root_) {
    stats_.n_nodes = count_nodes(root_.get());
    stats_.tree_depth = get_max_depth(root_.get());
    stats_.memory_bytes = compute_memory_usage();
  }

  auto end_time = std::chrono::steady_clock::now();
  stats_.build_time_ms = std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  is_built_ = true;
}

void BVHAccel::clear() {
  root_.reset();
  vertex_coords_.clear();
  primitive_aabbs_.clear();
  cell_indices_.clear();
  boundary_triangles_.clear();
  vertex_indices_.clear();
  vertex_to_cells_.clear();
  is_built_ = false;
  stats_ = SearchStats();
}

// ---- Tree construction ----

std::unique_ptr<BVHAccel::BVHNode> BVHAccel::build_recursive(
    std::vector<PrimitiveInfo>& primitives,
    int start, int end,
    int depth) {

  int n_primitives = end - start;

  // Create node and compute bounds
  auto node = std::make_unique<BVHNode>();
  node->depth = depth;
  node->bounds = compute_bounds(primitives, start, end);
  node->update_area();

  // Check for leaf node conditions
  if (n_primitives <= max_primitives_per_leaf_ || depth >= max_depth_) {
    return create_leaf(primitives, start, end, depth);
  }

  // Choose split method and axis
  int split_axis = -1;
  real_t split_cost = std::numeric_limits<real_t>::infinity();
  int mid = (start + end) / 2;

  if (build_method_ == BuildMethod::SAH) {
    // Use Surface Area Heuristic
    mid = find_sah_split(primitives, start, end, split_axis, split_cost);

    // Check if leaf is better than split
    real_t leaf_cost = intersection_cost_ * n_primitives;
    if (leaf_cost < split_cost || mid == start || mid == end) {
      return create_leaf(primitives, start, end, depth);
    }
  } else if (build_method_ == BuildMethod::Middle) {
    // Simple middle split
    search::AABB centroid_bounds = compute_centroid_bounds(primitives, start, end);
    auto extents = centroid_bounds.extents();

    // Choose longest axis
    split_axis = 0;
    if (extents[1] > extents[split_axis]) split_axis = 1;
    if (extents[2] > extents[split_axis]) split_axis = 2;

    // Partition at middle
    real_t split_pos = centroid_bounds.center()[split_axis];
    mid = partition_sah(primitives, start, end, split_axis, split_pos);

    if (mid == start || mid == end) {
      mid = (start + end) / 2;
      std::nth_element(primitives.begin() + start, primitives.begin() + mid,
                       primitives.begin() + end,
                       [split_axis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                         return a.centroid[split_axis] < b.centroid[split_axis];
                       });
    }
  } else {
    // Equal counts split
    search::AABB centroid_bounds = compute_centroid_bounds(primitives, start, end);
    auto extents = centroid_bounds.extents();

    split_axis = 0;
    if (extents[1] > extents[split_axis]) split_axis = 1;
    if (extents[2] > extents[split_axis]) split_axis = 2;

    mid = (start + end) / 2;
    std::nth_element(primitives.begin() + start, primitives.begin() + mid,
                     primitives.begin() + end,
                     [split_axis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                       return a.centroid[split_axis] < b.centroid[split_axis];
                     });
  }

  // Recursively build children
  node->left = build_recursive(primitives, start, mid, depth + 1);
  node->right = build_recursive(primitives, mid, end, depth + 1);

  return node;
}

int BVHAccel::find_sah_split(const std::vector<PrimitiveInfo>& primitives,
                              int start, int end,
                              int& split_axis,
                              real_t& split_cost) {

  int n_primitives = end - start;
  search::AABB bounds = compute_bounds(primitives, start, end);
  search::AABB centroid_bounds = compute_centroid_bounds(primitives, start, end);

  split_cost = std::numeric_limits<real_t>::infinity();
  int best_mid = (start + end) / 2;
  split_axis = -1;

  // Try each axis
  for (int axis = 0; axis < 3; ++axis) {
    // Skip degenerate axis
    if (centroid_bounds.max[axis] - centroid_bounds.min[axis] < 1e-6) {
      continue;
    }

    // Initialize buckets
    std::vector<SAHBucket> buckets(n_sah_buckets_);

    // Assign primitives to buckets
    for (int i = start; i < end; ++i) {
      real_t centroid = primitives[i].centroid[axis];
      int bucket_idx = static_cast<int>(n_sah_buckets_ *
          ((centroid - centroid_bounds.min[axis]) /
           (centroid_bounds.max[axis] - centroid_bounds.min[axis])));

      bucket_idx = std::min(bucket_idx, n_sah_buckets_ - 1);
      bucket_idx = std::max(bucket_idx, 0);

      buckets[bucket_idx].count++;
      if (buckets[bucket_idx].count == 1) {
        buckets[bucket_idx].bounds = primitives[i].bounds;
      } else {
        buckets[bucket_idx].bounds.expand(primitives[i].bounds);
      }
    }

    // Compute costs for each split
    for (int split = 1; split < n_sah_buckets_; ++split) {
      search::AABB left_bounds, right_bounds;
      int left_count = 0, right_count = 0;

      // Left side
      for (int i = 0; i < split; ++i) {
        if (buckets[i].count > 0) {
          if (left_count == 0) {
            left_bounds = buckets[i].bounds;
          } else {
            left_bounds.expand(buckets[i].bounds);
          }
          left_count += buckets[i].count;
        }
      }

      // Right side
      for (int i = split; i < n_sah_buckets_; ++i) {
        if (buckets[i].count > 0) {
          if (right_count == 0) {
            right_bounds = buckets[i].bounds;
          } else {
            right_bounds.expand(buckets[i].bounds);
          }
          right_count += buckets[i].count;
        }
      }

      // Compute SAH cost
      if (left_count > 0 && right_count > 0) {
        auto left_extents = left_bounds.extents();
        real_t left_area = 2.0 * (left_extents[0]*left_extents[1] +
                                  left_extents[1]*left_extents[2] +
                                  left_extents[2]*left_extents[0]);

        auto right_extents = right_bounds.extents();
        real_t right_area = 2.0 * (right_extents[0]*right_extents[1] +
                                   right_extents[1]*right_extents[2] +
                                   right_extents[2]*right_extents[0]);

        auto total_extents = bounds.extents();
        real_t total_area = 2.0 * (total_extents[0]*total_extents[1] +
                                   total_extents[1]*total_extents[2] +
                                   total_extents[2]*total_extents[0]);

        real_t cost = traversal_cost_ + intersection_cost_ *
                     (left_count * left_area / total_area +
                      right_count * right_area / total_area);

        if (cost < split_cost) {
          split_cost = cost;
          split_axis = axis;

          // Find actual split position
          real_t split_pos = centroid_bounds.min[axis] +
              split * (centroid_bounds.max[axis] - centroid_bounds.min[axis]) / n_sah_buckets_;

          // Find the actual middle index
          int count = 0;
          for (int i = start; i < end; ++i) {
            if (primitives[i].centroid[axis] < split_pos) {
              count++;
            }
          }
          best_mid = start + count;
        }
      }
    }
  }

  // Fallback if no good split found
  if (split_axis == -1) {
    auto extents = centroid_bounds.extents();
    split_axis = 0;
    if (extents[1] > extents[split_axis]) split_axis = 1;
    if (extents[2] > extents[split_axis]) split_axis = 2;
    best_mid = (start + end) / 2;
  }

  return best_mid;
}

int BVHAccel::partition_sah(std::vector<PrimitiveInfo>& primitives,
                            int start, int end,
                            int axis,
                            real_t split_pos) {

  auto it = std::partition(primitives.begin() + start, primitives.begin() + end,
                          [axis, split_pos](const PrimitiveInfo& p) {
                            return p.centroid[axis] < split_pos;
                          });

  return static_cast<int>(it - primitives.begin());
}

std::unique_ptr<BVHAccel::BVHNode> BVHAccel::create_leaf(
    const std::vector<PrimitiveInfo>& primitives,
    int start, int end,
    int depth) {

  auto node = std::make_unique<BVHNode>();
  node->is_leaf = true;
  node->depth = depth;
  node->bounds = compute_bounds(primitives, start, end);
  node->update_area();

  // Store primitive indices
  node->primitive_indices.reserve(end - start);
  for (int i = start; i < end; ++i) {
    node->primitive_indices.push_back(primitives[i].primitive_index);
  }

  return node;
}

search::AABB BVHAccel::compute_bounds(const std::vector<PrimitiveInfo>& primitives,
                                      int start, int end) const {
  search::AABB bounds = primitives[start].bounds;
  for (int i = start + 1; i < end; ++i) {
    bounds.expand(primitives[i].bounds);
  }
  return bounds;
}

search::AABB BVHAccel::compute_centroid_bounds(const std::vector<PrimitiveInfo>& primitives,
                                               int start, int end) const {
  search::AABB bounds(primitives[start].centroid, primitives[start].centroid);
  for (int i = start + 1; i < end; ++i) {
    bounds.expand(primitives[i].centroid);
  }
  return bounds;
}

// ---- Point location ----

PointLocateResult BVHAccel::locate_point(const MeshBase& mesh,
                                         const std::array<real_t,3>& point,
                                         index_t hint_cell) const {
  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;

  if (!is_built_ || !root_) {
    return result;
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  // Check hint first
  if (hint_cell >= 0 && hint_cell < mesh.n_cells()) {
    if (SearchBuilders::point_in_cell(mesh, vertex_coords_, hint_cell, point,
                                      result.parametric_coords)) {
      result.found = true;
      result.cell_id = hint_cell;

      auto end_time = std::chrono::steady_clock::now();
      stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
          end_time - start_time).count();
      return result;
    }
  }

  // Traverse BVH
  locate_point_recursive(root_.get(), mesh, point, result);

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return result;
}

void BVHAccel::locate_point_recursive(const BVHNode* node,
                                      const MeshBase& mesh,
                                      const std::array<real_t,3>& point,
                                      PointLocateResult& result) const {
  if (!node || !node->bounds.contains(point)) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check primitives in leaf
    for (index_t prim_idx : node->primitive_indices) {
      index_t cell_id = cell_indices_.empty() ? prim_idx : cell_indices_[prim_idx];

      if (SearchBuilders::point_in_cell(mesh, vertex_coords_, cell_id, point,
                                        result.parametric_coords)) {
        result.found = true;
        result.cell_id = cell_id;
        return;  // Found the containing cell
      }
    }
  } else {
    // Traverse children
    locate_point_recursive(node->left.get(), mesh, point, result);
    if (result.found) return;

    locate_point_recursive(node->right.get(), mesh, point, result);
  }
}

std::vector<PointLocateResult> BVHAccel::locate_points(
    const MeshBase& mesh,
    const std::vector<std::array<real_t,3>>& points) const {

  std::vector<PointLocateResult> results;
  results.reserve(points.size());

  for (const auto& point : points) {
    results.push_back(locate_point(mesh, point));
  }

  return results;
}

// ---- Nearest neighbor ----

std::pair<index_t, real_t> BVHAccel::nearest_vertex(
    const MeshBase& mesh,
    const std::array<real_t,3>& point) const {

  if (!is_built_ || vertex_coords_.empty()) {
    return {-1, std::numeric_limits<real_t>::infinity()};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  index_t best_idx = -1;
  real_t best_dist_sq = std::numeric_limits<real_t>::infinity();

  nearest_vertex_recursive(root_.get(), point, best_idx, best_dist_sq);

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return {best_idx, std::sqrt(best_dist_sq)};
}

void BVHAccel::nearest_vertex_recursive(const BVHNode* node,
                                        const std::array<real_t,3>& point,
                                        index_t& best_idx,
                                        real_t& best_dist_sq) const {
  if (!node) return;

  // Check if this node can contain a closer point
  real_t node_dist_sq = distance_to_node(node, point);
  if (node_dist_sq >= best_dist_sq) {
    return;  // Prune this branch
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // For vertex queries, need to check actual vertices
    // This assumes primitives are cells, need to get vertices from cells
    for (index_t prim_idx : node->primitive_indices) {
      if (!cell_indices_.empty()) {
        // Get vertices of this cell
        index_t cell_id = cell_indices_[prim_idx];
        // Note: This is a simplified approach, actual implementation would need
        // proper vertex extraction from cells
      }
    }

    // Direct vertex check if we stored vertices
    for (index_t vid : vertex_indices_) {
      const auto& v = vertex_coords_[vid];
      real_t dist_sq = search::norm_squared(search::subtract(v, point));
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_idx = vid;
      }
    }
  } else {
    // Visit children in order of distance
    real_t left_dist = distance_to_node(node->left.get(), point);
    real_t right_dist = distance_to_node(node->right.get(), point);

    if (left_dist < right_dist) {
      nearest_vertex_recursive(node->left.get(), point, best_idx, best_dist_sq);
      if (right_dist < best_dist_sq) {
        nearest_vertex_recursive(node->right.get(), point, best_idx, best_dist_sq);
      }
    } else {
      nearest_vertex_recursive(node->right.get(), point, best_idx, best_dist_sq);
      if (left_dist < best_dist_sq) {
        nearest_vertex_recursive(node->left.get(), point, best_idx, best_dist_sq);
      }
    }
  }
}

std::vector<std::pair<index_t, real_t>> BVHAccel::k_nearest_vertices(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    size_t k) const {

  if (!is_built_ || vertex_coords_.empty() || k == 0) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  // Max heap to track k nearest
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

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void BVHAccel::k_nearest_recursive(const BVHNode* node,
                                   const std::array<real_t,3>& point,
                                   size_t k,
                                   std::priority_queue<std::pair<real_t, index_t>>& max_heap) const {
  if (!node) return;

  // Prune if node is too far
  real_t node_dist_sq = distance_to_node(node, point);
  if (!max_heap.empty() && max_heap.size() >= k && node_dist_sq >= max_heap.top().first) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check vertices in leaf
    for (index_t vid : vertex_indices_) {
      const auto& v = vertex_coords_[vid];
      real_t dist_sq = search::norm_squared(search::subtract(v, point));

      if (max_heap.size() < k) {
        max_heap.push({dist_sq, vid});
      } else if (dist_sq < max_heap.top().first) {
        max_heap.pop();
        max_heap.push({dist_sq, vid});
      }
    }
  } else {
    // Visit children in order of distance
    real_t left_dist = distance_to_node(node->left.get(), point);
    real_t right_dist = distance_to_node(node->right.get(), point);

    if (left_dist < right_dist) {
      k_nearest_recursive(node->left.get(), point, k, max_heap);
      if (max_heap.size() < k || right_dist < max_heap.top().first) {
        k_nearest_recursive(node->right.get(), point, k, max_heap);
      }
    } else {
      k_nearest_recursive(node->right.get(), point, k, max_heap);
      if (max_heap.size() < k || left_dist < max_heap.top().first) {
        k_nearest_recursive(node->left.get(), point, k, max_heap);
      }
    }
  }
}

std::vector<index_t> BVHAccel::vertices_in_radius(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    real_t radius) const {

  if (!is_built_ || vertex_coords_.empty()) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  std::vector<index_t> results;
  real_t radius_sq = radius * radius;

  radius_search_recursive(root_.get(), point, radius_sq, results);

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void BVHAccel::radius_search_recursive(const BVHNode* node,
                                       const std::array<real_t,3>& point,
                                       real_t radius_sq,
                                       std::vector<index_t>& results) const {
  if (!node) return;

  // Check if sphere overlaps node
  real_t node_dist_sq = distance_to_node(node, point);
  if (node_dist_sq > radius_sq) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check vertices in leaf
    for (index_t vid : vertex_indices_) {
      const auto& v = vertex_coords_[vid];
      real_t dist_sq = search::norm_squared(search::subtract(v, point));
      if (dist_sq <= radius_sq) {
        results.push_back(vid);
      }
    }
  } else {
    radius_search_recursive(node->left.get(), point, radius_sq, results);
    radius_search_recursive(node->right.get(), point, radius_sq, results);
  }
}

// ---- Ray intersection ----

RayIntersectResult BVHAccel::intersect_ray(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  RayIntersectResult result;
  result.hit = false;
  result.distance = max_distance;

  if (!is_built_ || !root_ || boundary_triangles_.empty()) {
    return result;
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  search::Ray ray(origin, direction);
  real_t t_min = max_distance;

  intersect_ray_node(root_.get(), ray, result, t_min);

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return result;
}

bool BVHAccel::intersect_ray_node(const BVHNode* node,
                                  const search::Ray& ray,
                                  RayIntersectResult& result,
                                  real_t& t_min) const {
  if (!node) return false;

  // Check ray-AABB intersection
  real_t t_near, t_far;
  if (!ray_intersects_box(ray, node->bounds, t_near, t_far)) {
    return false;
  }

  // Early exit if intersection is beyond current best
  if (t_near > t_min) {
    return false;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    bool hit = false;

    // Test ray against triangles in leaf
    for (index_t prim_idx : node->primitive_indices) {
      const auto& tri = boundary_triangles_[prim_idx];

      real_t t;
      std::array<real_t,3> hit_point;
      std::array<real_t,2> bary;

      if (ray_triangle_intersection(ray, tri, t, hit_point, bary)) {
        if (t < t_min) {
          t_min = t;
          result.hit = true;
          result.distance = t;
          result.hit_point = hit_point;
          result.normal = tri.normal;
          result.face_id = tri.face_id;
          result.barycentric = {bary[0], bary[1], 1.0 - bary[0] - bary[1]};
          hit = true;
        }
      }
    }

    return hit;
  } else {
    // Traverse children, visiting closer one first
    real_t left_near, left_far, right_near, right_far;
    bool left_hit = ray_intersects_box(ray, node->left->bounds, left_near, left_far);
    bool right_hit = ray_intersects_box(ray, node->right->bounds, right_near, right_far);

    bool hit = false;

    if (left_hit && right_hit) {
      // Both children intersected, visit closer one first
      if (left_near < right_near) {
        hit = intersect_ray_node(node->left.get(), ray, result, t_min) || hit;
        if (right_near < t_min) {
          hit = intersect_ray_node(node->right.get(), ray, result, t_min) || hit;
        }
      } else {
        hit = intersect_ray_node(node->right.get(), ray, result, t_min) || hit;
        if (left_near < t_min) {
          hit = intersect_ray_node(node->left.get(), ray, result, t_min) || hit;
        }
      }
    } else if (left_hit) {
      hit = intersect_ray_node(node->left.get(), ray, result, t_min);
    } else if (right_hit) {
      hit = intersect_ray_node(node->right.get(), ray, result, t_min);
    }

    return hit;
  }
}

std::vector<RayIntersectResult> BVHAccel::intersect_ray_all(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  std::vector<RayIntersectResult> results;

  if (!is_built_ || !root_ || boundary_triangles_.empty()) {
    return results;
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  search::Ray ray(origin, direction);
  ray.max_t = max_distance;

  intersect_ray_all_recursive(root_.get(), ray, results);

  // Sort by distance
  std::sort(results.begin(), results.end(),
           [](const RayIntersectResult& a, const RayIntersectResult& b) {
             return a.distance < b.distance;
           });

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void BVHAccel::intersect_ray_all_recursive(const BVHNode* node,
                                           const search::Ray& ray,
                                           std::vector<RayIntersectResult>& results) const {
  if (!node) return;

  // Check ray-AABB intersection
  real_t t_near, t_far;
  if (!ray_intersects_box(ray, node->bounds, t_near, t_far)) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Test all triangles in leaf
    for (index_t prim_idx : node->primitive_indices) {
      const auto& tri = boundary_triangles_[prim_idx];

      real_t t;
      std::array<real_t,3> hit_point;
      std::array<real_t,2> bary;

      if (ray_triangle_intersection(ray, tri, t, hit_point, bary)) {
        if (t <= ray.max_t) {
          RayIntersectResult result;
          result.hit = true;
          result.distance = t;
          result.hit_point = hit_point;
          result.normal = tri.normal;
          result.face_id = tri.face_id;
          result.barycentric = {bary[0], bary[1], 1.0 - bary[0] - bary[1]};
          results.push_back(result);
        }
      }
    }
  } else {
    // Traverse both children
    intersect_ray_all_recursive(node->left.get(), ray, results);
    intersect_ray_all_recursive(node->right.get(), ray, results);
  }
}

// ---- Region queries ----

std::vector<index_t> BVHAccel::cells_in_box(
    const MeshBase& mesh,
    const std::array<real_t,3>& box_min,
    const std::array<real_t,3>& box_max) const {

  if (!is_built_ || !root_) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  search::AABB query_box(box_min, box_max);
  std::unordered_set<index_t> result_set;

  box_search_recursive(root_.get(), query_box, result_set);

  std::vector<index_t> results(result_set.begin(), result_set.end());

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void BVHAccel::box_search_recursive(const BVHNode* node,
                                    const search::AABB& box,
                                    std::unordered_set<index_t>& results) const {
  if (!node || !node->bounds.overlaps(box)) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check primitives in leaf
    for (index_t prim_idx : node->primitive_indices) {
      index_t cell_id = cell_indices_.empty() ? prim_idx : cell_indices_[prim_idx];

      if (prim_idx < primitive_aabbs_.size() &&
          primitive_aabbs_[prim_idx].overlaps(box)) {
        results.insert(cell_id);
      }
    }
  } else {
    box_search_recursive(node->left.get(), box, results);
    box_search_recursive(node->right.get(), box, results);
  }
}

std::vector<index_t> BVHAccel::cells_in_sphere(
    const MeshBase& mesh,
    const std::array<real_t,3>& center,
    real_t radius) const {

  if (!is_built_ || !root_) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  std::unordered_set<index_t> result_set;

  sphere_search_recursive(root_.get(), center, radius, result_set);

  std::vector<index_t> results(result_set.begin(), result_set.end());

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void BVHAccel::sphere_search_recursive(const BVHNode* node,
                                       const std::array<real_t,3>& center,
                                       real_t radius,
                                       std::unordered_set<index_t>& results) const {
  if (!node) return;

  // Check if sphere overlaps node bounds
  if (distance_to_node(node, center) > radius) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check primitives in leaf
    real_t radius_sq = radius * radius;

    for (index_t prim_idx : node->primitive_indices) {
      index_t cell_id = cell_indices_.empty() ? prim_idx : cell_indices_[prim_idx];

      // Check if cell AABB overlaps sphere
      if (prim_idx < primitive_aabbs_.size()) {
        auto closest = primitive_aabbs_[prim_idx].closest_point(center);
        real_t dist_sq = search::norm_squared(search::subtract(closest, center));
        if (dist_sq <= radius_sq) {
          results.insert(cell_id);
        }
      }
    }
  } else {
    sphere_search_recursive(node->left.get(), center, radius, results);
    sphere_search_recursive(node->right.get(), center, radius, results);
  }
}

// ---- BVH-specific methods ----

void BVHAccel::refit(const MeshBase& mesh) {
  if (!is_built_ || !root_) return;

  auto start_time = std::chrono::steady_clock::now();

  // Update vertex coordinates
  if (built_cfg_ == Configuration::Reference) {
    vertex_coords_ = SearchBuilders::extract_vertex_coords(mesh);
  } else {
    vertex_coords_ = SearchBuilders::extract_deformed_coords(mesh);
  }

  // Update primitive bounds
  if (!boundary_triangles_.empty()) {
    // Update triangle vertices
    boundary_triangles_ = SearchBuilders::extract_boundary_triangles(mesh, vertex_coords_);
  } else {
    // Update cell AABBs
    primitive_aabbs_ = SearchBuilders::compute_cell_aabbs(mesh, vertex_coords_);
  }

  // Refit tree nodes bottom-up
  refit_recursive(root_.get());

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();
}

void BVHAccel::refit_recursive(BVHNode* node) {
  if (!node) return;

  if (node->is_leaf) {
    // Recompute bounds from primitives
    bool first = true;
    for (index_t prim_idx : node->primitive_indices) {
      search::AABB prim_bounds;

      if (!boundary_triangles_.empty() && prim_idx < boundary_triangles_.size()) {
        const auto& tri = boundary_triangles_[prim_idx];
        prim_bounds = search::AABB(tri.vertices[0], tri.vertices[0]);
        prim_bounds.expand(tri.vertices[1]);
        prim_bounds.expand(tri.vertices[2]);
      } else if (prim_idx < primitive_aabbs_.size()) {
        prim_bounds = primitive_aabbs_[prim_idx];
      }

      if (first) {
        node->bounds = prim_bounds;
        first = false;
      } else {
        node->bounds.expand(prim_bounds);
      }
    }
  } else {
    // Refit children first
    refit_recursive(node->left.get());
    refit_recursive(node->right.get());

    // Update node bounds from children
    node->bounds = node->left->bounds;
    node->bounds.expand(node->right->bounds);
  }

  node->update_area();
}

real_t BVHAccel::compute_sah_cost() const {
  if (!root_) return 0.0;
  return compute_sah_cost_recursive(root_.get());
}

real_t BVHAccel::compute_sah_cost_recursive(const BVHNode* node) const {
  if (!node) return 0.0;

  if (node->is_leaf) {
    return intersection_cost_ * node->primitive_indices.size();
  } else {
    real_t left_cost = compute_sah_cost_recursive(node->left.get());
    real_t right_cost = compute_sah_cost_recursive(node->right.get());

    real_t left_area = node->left->area;
    real_t right_area = node->right->area;
    real_t total_area = node->area;

    return traversal_cost_ +
           (left_area / total_area) * left_cost +
           (right_area / total_area) * right_cost;
  }
}

// ---- Helper methods ----

real_t BVHAccel::distance_to_node(const BVHNode* node,
                                  const std::array<real_t,3>& point) const {
  if (!node) return std::numeric_limits<real_t>::infinity();

  auto closest = node->bounds.closest_point(point);
  return search::norm_squared(search::subtract(closest, point));
}

bool BVHAccel::ray_intersects_box(const search::Ray& ray,
                                  const search::AABB& box,
                                  real_t& t_near,
                                  real_t& t_far) const {
  t_near = 0.0;
  t_far = ray.max_t;

  for (int i = 0; i < 3; ++i) {
    real_t inv_dir = 1.0 / ray.direction[i];
    real_t t0 = (box.min[i] - ray.origin[i]) * inv_dir;
    real_t t1 = (box.max[i] - ray.origin[i]) * inv_dir;

    if (inv_dir < 0.0) {
      std::swap(t0, t1);
    }

    t_near = std::max(t_near, t0);
    t_far = std::min(t_far, t1);

    if (t_near > t_far) {
      return false;
    }
  }

  return true;
}

bool BVHAccel::ray_triangle_intersection(const search::Ray& ray,
                                         const BoundaryTriangle& tri,
                                         real_t& t,
                                         std::array<real_t,3>& hit_point,
                                         std::array<real_t,2>& bary) const {
  // Möller–Trumbore intersection algorithm
  const auto& v0 = tri.vertices[0];
  const auto& v1 = tri.vertices[1];
  const auto& v2 = tri.vertices[2];

  auto edge1 = search::subtract(v1, v0);
  auto edge2 = search::subtract(v2, v0);
  auto h = search::cross(ray.direction, edge2);
  real_t a = search::dot(edge1, h);

  if (std::abs(a) < 1e-10) {
    return false;  // Ray parallel to triangle
  }

  real_t f = 1.0 / a;
  auto s = search::subtract(ray.origin, v0);
  real_t u = f * search::dot(s, h);

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  auto q = search::cross(s, edge1);
  real_t v = f * search::dot(ray.direction, q);

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  t = f * search::dot(edge2, q);

  if (t > 0.0 && t <= ray.max_t) {
    bary[0] = u;
    bary[1] = v;
    hit_point = ray.point_at(t);
    return true;
  }

  return false;
}

size_t BVHAccel::count_nodes(const BVHNode* node) const {
  if (!node) return 0;
  return 1 + count_nodes(node->left.get()) + count_nodes(node->right.get());
}

int BVHAccel::get_max_depth(const BVHNode* node) const {
  if (!node) return 0;
  if (node->is_leaf) return 1;
  return 1 + std::max(get_max_depth(node->left.get()),
                     get_max_depth(node->right.get()));
}

size_t BVHAccel::compute_memory_usage() const {
  size_t total = 0;

  // Tree nodes
  if (root_) {
    total += count_nodes(root_.get()) * sizeof(BVHNode);
  }

  // Cached data
  total += vertex_coords_.size() * sizeof(std::array<real_t,3>);
  total += primitive_aabbs_.size() * sizeof(search::AABB);
  total += boundary_triangles_.size() * sizeof(BoundaryTriangle);
  total += vertex_indices_.size() * sizeof(index_t);
  total += cell_indices_.size() * sizeof(index_t);

  for (const auto& vc : vertex_to_cells_) {
    total += vc.size() * sizeof(index_t);
  }

  return total;
}

void BVHAccel::flatten_tree() {
  // Optional optimization: convert tree to linear array for better cache performance
  // This would involve creating a linear array of nodes and updating traversal
  // to use indices instead of pointers
}

} // namespace svmp