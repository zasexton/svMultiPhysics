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

#include "OctreeAccel.h"
#include "SearchBuilders.h"
#include "../Core/MeshBase.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stack>
#include <limits>

namespace svmp {

// ---- Building ----

void OctreeAccel::build(const MeshBase& mesh,
                        Configuration cfg,
                        const MeshSearch::SearchConfig& config) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Clear existing structure
  clear();

  // Store configuration
  built_cfg_ = cfg;
  max_depth_ = config.max_depth;
  min_entities_per_leaf_ = config.min_entities_per_leaf;

  // Set max entities per leaf based on config
  max_entities_per_leaf_ = min_entities_per_leaf_ * 10;

  // Extract mesh data
  vertex_coords_ = search::SearchBuilders::extract_vertex_coords(mesh, cfg);
  cell_aabbs_ = search::SearchBuilders::compute_all_cell_aabbs(mesh, cfg);

  if (vertex_coords_.empty() && cell_aabbs_.empty()) {
    is_built_ = true;
    return;
  }

  // Build boundary triangles for ray intersection
  auto boundary_tris = search::SearchBuilders::triangulate_boundary(mesh, cfg);
  boundary_triangles_.clear();
  boundary_triangles_.reserve(boundary_tris.size());
  for (const auto& tri : boundary_tris) {
    BoundaryTriangle bt;
    bt.vertices = tri.vertices;
    bt.face_id = tri.face_id;
    boundary_triangles_.push_back(bt);
  }

  // Compute overall bounds
  search::AABB tree_bounds = search::SearchBuilders::compute_mesh_aabb(mesh, cfg);

  // Add small padding to avoid numerical issues
  for (int i = 0; i < 3; ++i) {
    real_t padding = (tree_bounds.max[i] - tree_bounds.min[i]) * 0.01;
    tree_bounds.min[i] -= padding;
    tree_bounds.max[i] += padding;
  }

  // Create root node
  root_ = std::make_unique<OctreeNode>(tree_bounds, 0);

  // Create initial entity lists
  std::vector<index_t> all_cells(cell_aabbs_.size());
  std::iota(all_cells.begin(), all_cells.end(), 0);

  std::vector<index_t> all_vertices(vertex_coords_.size());
  std::iota(all_vertices.begin(), all_vertices.end(), 0);

  // Build tree recursively
  build_node(root_.get(), all_cells, all_vertices);

  is_built_ = true;

  // Update statistics
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  stats_.build_time_ms = duration.count();
  stats_.n_entities = vertex_coords_.size() + cell_aabbs_.size();
  stats_.tree_depth = get_max_depth(root_.get());
  stats_.memory_bytes = compute_memory_usage();
}

void OctreeAccel::clear() {
  root_.reset();
  vertex_coords_.clear();
  cell_aabbs_.clear();
  boundary_triangles_.clear();
  is_built_ = false;
  stats_ = SearchStats();
}

// ---- Tree construction helpers ----

void OctreeAccel::build_node(OctreeNode* node,
                             const std::vector<index_t>& cell_indices,
                             const std::vector<index_t>& vertex_indices) {

  size_t n_cells = cell_indices.size();
  size_t n_vertices = vertex_indices.size();

  // Check if this should be a leaf
  if (!should_subdivide(node, n_cells, n_vertices)) {
    node->is_leaf = true;
    node->cell_indices = cell_indices;
    node->vertex_indices = vertex_indices;
    return;
  }

  // Distribute entities to children
  std::array<std::vector<index_t>, 8> child_cells;
  std::array<std::vector<index_t>, 8> child_vertices;
  distribute_entities(node, cell_indices, vertex_indices,
                     child_cells, child_vertices);

  // Create and build child nodes
  for (int octant = 0; octant < 8; ++octant) {
    if (!child_cells[octant].empty() || !child_vertices[octant].empty()) {
      auto child_bounds = node->get_child_bounds(octant);
      node->children[octant] = std::make_unique<OctreeNode>(
          child_bounds, node->depth + 1);

      build_node(node->children[octant].get(),
                child_cells[octant],
                child_vertices[octant]);
    }
  }
}

bool OctreeAccel::should_subdivide(const OctreeNode* node,
                                   size_t n_cells,
                                   size_t n_vertices) const {
  // Don't subdivide if at max depth
  if (node->depth >= max_depth_) {
    return false;
  }

  // Don't subdivide if too few entities
  size_t total_entities = n_cells + n_vertices;
  if (total_entities <= static_cast<size_t>(min_entities_per_leaf_)) {
    return false;
  }

  // Subdivide if too many entities
  if (total_entities > static_cast<size_t>(max_entities_per_leaf_)) {
    return true;
  }

  // Check spatial extent - don't subdivide tiny nodes
  auto extents = node->bounds.extents();
  real_t min_extent = std::min({extents[0], extents[1], extents[2]});
  if (min_extent < 1e-6) {
    return false;
  }

  return true;
}

void OctreeAccel::distribute_entities(const OctreeNode* parent,
                                      const std::vector<index_t>& cell_indices,
                                      const std::vector<index_t>& vertex_indices,
                                      std::array<std::vector<index_t>, 8>& child_cells,
                                      std::array<std::vector<index_t>, 8>& child_vertices) {

  // Clear child arrays
  for (int i = 0; i < 8; ++i) {
    child_cells[i].clear();
    child_vertices[i].clear();
  }

  // Distribute cells based on their AABBs
  for (index_t cell_id : cell_indices) {
    const auto& aabb = cell_aabbs_[cell_id];

    // Check which octants the cell overlaps
    for (int octant = 0; octant < 8; ++octant) {
      auto child_bounds = parent->get_child_bounds(octant);
      if (aabb.overlaps(child_bounds)) {
        child_cells[octant].push_back(cell_id);
      }
    }
  }

  // Distribute vertices based on position
  for (index_t vertex_id : vertex_indices) {
    const auto& pos = vertex_coords_[vertex_id];
    int octant = parent->get_octant(pos);
    child_vertices[octant].push_back(vertex_id);
  }
}

// ---- Point location ----

PointLocateResult OctreeAccel::locate_point(const MeshBase& mesh,
                                           const std::array<real_t,3>& point,
                                           index_t hint_cell) const {
  if (!is_built_) {
    PointLocateResult result;
    result.found = false;
    result.cell_id = -1;
    return result;
  }

  stats_.query_count++;

  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;

  // Try hint cell first
  if (hint_cell >= 0 && hint_cell < static_cast<index_t>(cell_aabbs_.size())) {
    if (cell_aabbs_[hint_cell].contains(point)) {
      auto vertex_coords = search::SearchBuilders::get_cell_vertex_coords(
          mesh, hint_cell, built_cfg_);
      auto shape = mesh.cell_shape(hint_cell);

      if (search::point_in_cell(point, shape, vertex_coords)) {
        result.found = true;
        result.cell_id = hint_cell;
        result.xi = search::SearchBuilders::compute_parametric_coords(
            mesh, hint_cell, point, built_cfg_);
        stats_.hit_count++;
        return result;
      }
    }
  }

  // Search through octree
  locate_point_recursive(root_.get(), mesh, point, result);

  if (result.found) {
    stats_.hit_count++;
  }

  return result;
}

void OctreeAccel::locate_point_recursive(const OctreeNode* node,
                                        const MeshBase& mesh,
                                        const std::array<real_t,3>& point,
                                        PointLocateResult& result) const {
  if (!node || !node->bounds.contains(point)) {
    return;
  }

  if (node->is_leaf) {
    // Check cells in this leaf
    for (index_t cell_id : node->cell_indices) {
      if (cell_aabbs_[cell_id].contains(point)) {
        auto vertex_coords = search::SearchBuilders::get_cell_vertex_coords(
            mesh, cell_id, built_cfg_);
        auto shape = mesh.cell_shape(cell_id);

        if (search::point_in_cell(point, shape, vertex_coords)) {
          result.found = true;
          result.cell_id = cell_id;
          result.xi = search::SearchBuilders::compute_parametric_coords(
              mesh, cell_id, point, built_cfg_);
          return;
        }
      }
    }
  } else {
    // Recurse to appropriate child
    int octant = node->get_octant(point);
    if (node->children[octant]) {
      locate_point_recursive(node->children[octant].get(), mesh, point, result);
    }
  }
}

std::vector<PointLocateResult> OctreeAccel::locate_points(
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

std::pair<index_t, real_t> OctreeAccel::nearest_vertex(
    const MeshBase& mesh,
    const std::array<real_t,3>& point) const {

  if (!is_built_ || vertex_coords_.empty()) {
    return {INVALID_INDEX, std::numeric_limits<real_t>::max()};
  }

  stats_.query_count++;

  index_t best_idx = INVALID_INDEX;
  real_t best_dist_sq = std::numeric_limits<real_t>::max();

  nearest_vertex_recursive(root_.get(), point, best_idx, best_dist_sq);

  if (best_idx != INVALID_INDEX) {
    stats_.hit_count++;
  }

  return {best_idx, std::sqrt(best_dist_sq)};
}

void OctreeAccel::nearest_vertex_recursive(const OctreeNode* node,
                                          const std::array<real_t,3>& point,
                                          index_t& best_idx,
                                          real_t& best_dist_sq) const {
  if (!node) return;

  // Check if this node could contain a closer point
  real_t dist_to_node = distance_to_node(node, point);
  if (dist_to_node * dist_to_node >= best_dist_sq) {
    return;
  }

  if (node->is_leaf) {
    // Check all vertices in leaf
    for (index_t idx : node->vertex_indices) {
      const auto& v = vertex_coords_[idx];
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
  } else {
    // Visit children in order of distance
    std::array<std::pair<real_t, int>, 8> child_dists;
    int n_children = 0;

    for (int octant = 0; octant < 8; ++octant) {
      if (node->children[octant]) {
        real_t dist = distance_to_node(node->children[octant].get(), point);
        child_dists[n_children++] = {dist, octant};
      }
    }

    // Sort by distance
    std::sort(child_dists.begin(), child_dists.begin() + n_children);

    // Visit children in order
    for (int i = 0; i < n_children; ++i) {
      int octant = child_dists[i].second;
      nearest_vertex_recursive(node->children[octant].get(),
                             point, best_idx, best_dist_sq);
    }
  }
}

std::vector<std::pair<index_t, real_t>> OctreeAccel::k_nearest_vertices(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    size_t k) const {

  if (!is_built_ || vertex_coords_.empty()) {
    return {};
  }

  stats_.query_count++;

  // Max heap to keep k smallest distances
  std::priority_queue<std::pair<real_t, index_t>> max_heap;

  k_nearest_search(point, k, max_heap);

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

void OctreeAccel::k_nearest_search(const std::array<real_t,3>& point,
                                   size_t k,
                                   std::priority_queue<std::pair<real_t, index_t>>& max_heap) const {

  // Priority queue for traversal (min heap by distance)
  std::priority_queue<TraversalEntry> traversal_queue;

  if (root_) {
    traversal_queue.push({root_.get(), 0});
  }

  while (!traversal_queue.empty()) {
    auto entry = traversal_queue.top();
    traversal_queue.pop();

    const OctreeNode* node = entry.node;
    real_t node_dist = entry.priority;

    // Skip if node can't contain closer points
    if (max_heap.size() == k && node_dist * node_dist >= max_heap.top().first) {
      continue;
    }

    if (node->is_leaf) {
      // Check all vertices in leaf
      for (index_t idx : node->vertex_indices) {
        const auto& v = vertex_coords_[idx];
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
    } else {
      // Add children to traversal queue
      for (int octant = 0; octant < 8; ++octant) {
        if (node->children[octant]) {
          real_t child_dist = distance_to_node(node->children[octant].get(), point);
          traversal_queue.push({node->children[octant].get(), child_dist});
        }
      }
    }
  }
}

std::vector<index_t> OctreeAccel::vertices_in_radius(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    real_t radius) const {

  if (!is_built_ || vertex_coords_.empty()) {
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

void OctreeAccel::radius_search_recursive(const OctreeNode* node,
                                         const std::array<real_t,3>& point,
                                         real_t radius_sq,
                                         std::vector<index_t>& results) const {
  if (!node) return;

  // Check if node intersects search sphere
  if (!search::aabb_sphere_overlap(node->bounds, point, std::sqrt(radius_sq))) {
    return;
  }

  if (node->is_leaf) {
    // Check all vertices in leaf
    for (index_t idx : node->vertex_indices) {
      const auto& v = vertex_coords_[idx];
      real_t dist_sq = 0;
      for (int d = 0; d < 3; ++d) {
        real_t dx = v[d] - point[d];
        dist_sq += dx * dx;
      }

      if (dist_sq <= radius_sq) {
        results.push_back(idx);
      }
    }
  } else {
    // Recurse to all overlapping children
    for (int octant = 0; octant < 8; ++octant) {
      if (node->children[octant]) {
        radius_search_recursive(node->children[octant].get(),
                              point, radius_sq, results);
      }
    }
  }
}

// ---- Ray intersection ----

RayIntersectResult OctreeAccel::intersect_ray(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  if (!is_built_ || boundary_triangles_.empty()) {
    RayIntersectResult result;
    result.found = false;
    result.face_id = -1;
    result.t = -1.0;
    return result;
  }

  stats_.query_count++;

  search::Ray ray(origin, search::normalize3(direction), 0.0, max_distance);

  // Traverse octree and collect leaf nodes hit by ray
  std::vector<const OctreeNode*> hit_leaves;
  ray_traverse(root_.get(), ray, hit_leaves);

  // Test triangles in hit leaves
  RayIntersectResult best_result;
  best_result.found = false;
  best_result.t = max_distance;

  for (const OctreeNode* leaf : hit_leaves) {
    RayIntersectResult node_result;
    if (ray_node_intersection(leaf, ray, node_result)) {
      if (node_result.t < best_result.t) {
        best_result = node_result;
      }
    }
  }

  if (best_result.found) {
    stats_.hit_count++;
  }

  return best_result;
}

std::vector<RayIntersectResult> OctreeAccel::intersect_ray_all(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  if (!is_built_ || boundary_triangles_.empty()) {
    return {};
  }

  stats_.query_count++;

  search::Ray ray(origin, search::normalize3(direction), 0.0, max_distance);

  // Traverse octree and collect leaf nodes hit by ray
  std::vector<const OctreeNode*> hit_leaves;
  ray_traverse(root_.get(), ray, hit_leaves);

  // Collect all intersections
  std::vector<RayIntersectResult> results;

  std::unordered_set<size_t> tested_triangles;
  for (const OctreeNode* leaf : hit_leaves) {
    // Test boundary triangles associated with cells in this leaf
    for (index_t cell_id : leaf->cell_indices) {
      // This is simplified - would need proper cell-to-triangle mapping
      for (size_t tri_idx = 0; tri_idx < boundary_triangles_.size(); ++tri_idx) {
        if (tested_triangles.count(tri_idx) > 0) continue;
        tested_triangles.insert(tri_idx);

        const auto& tri = boundary_triangles_[tri_idx];
        real_t t;
        if (search::ray_triangle_intersect(ray, tri.vertices[0],
                                          tri.vertices[1],
                                          tri.vertices[2], t)) {
          RayIntersectResult result;
          result.found = true;
          result.face_id = tri.face_id;
          result.t = t;
          result.point = ray.point_at(t);
          results.push_back(result);
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

void OctreeAccel::ray_traverse(const OctreeNode* node,
                               const search::Ray& ray,
                               std::vector<const OctreeNode*>& leaves) const {
  if (!node) return;

  real_t t_near, t_far;
  if (!ray_intersects_node(node, ray, t_near, t_far)) {
    return;
  }

  if (node->is_leaf) {
    leaves.push_back(node);
  } else {
    // Traverse children
    for (int octant = 0; octant < 8; ++octant) {
      if (node->children[octant]) {
        ray_traverse(node->children[octant].get(), ray, leaves);
      }
    }
  }
}

bool OctreeAccel::ray_node_intersection(const OctreeNode* node,
                                        const search::Ray& ray,
                                        RayIntersectResult& result) const {
  result.found = false;

  // Test triangles associated with cells in this node
  // This is simplified - a real implementation would maintain triangle-to-node mapping
  for (const auto& tri : boundary_triangles_) {
    real_t t;
    if (search::ray_triangle_intersect(ray, tri.vertices[0],
                                      tri.vertices[1],
                                      tri.vertices[2], t)) {
      if (!result.found || t < result.t) {
        result.found = true;
        result.face_id = tri.face_id;
        result.t = t;
        result.point = ray.point_at(t);
      }
    }
  }

  return result.found;
}

// ---- Region queries ----

std::vector<index_t> OctreeAccel::cells_in_box(
    const MeshBase& mesh,
    const std::array<real_t,3>& box_min,
    const std::array<real_t,3>& box_max) const {

  if (!is_built_) {
    return {};
  }

  stats_.query_count++;

  search::AABB query_box(box_min, box_max);
  std::unordered_set<index_t> unique_cells;

  box_search_recursive(root_.get(), query_box, unique_cells);

  std::vector<index_t> results(unique_cells.begin(), unique_cells.end());

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

void OctreeAccel::box_search_recursive(const OctreeNode* node,
                                       const search::AABB& box,
                                       std::unordered_set<index_t>& results) const {
  if (!node || !node->bounds.overlaps(box)) {
    return;
  }

  if (node->is_leaf) {
    // Check cells in this node
    for (index_t cell_id : node->cell_indices) {
      if (cell_aabbs_[cell_id].overlaps(box)) {
        results.insert(cell_id);
      }
    }
  } else {
    // Recurse to overlapping children
    for (int octant = 0; octant < 8; ++octant) {
      if (node->children[octant]) {
        box_search_recursive(node->children[octant].get(), box, results);
      }
    }
  }
}

std::vector<index_t> OctreeAccel::cells_in_sphere(
    const MeshBase& mesh,
    const std::array<real_t,3>& center,
    real_t radius) const {

  if (!is_built_) {
    return {};
  }

  stats_.query_count++;

  std::unordered_set<index_t> unique_cells;

  sphere_search_recursive(root_.get(), center, radius, unique_cells);

  std::vector<index_t> results(unique_cells.begin(), unique_cells.end());

  if (!results.empty()) {
    stats_.hit_count++;
  }

  return results;
}

void OctreeAccel::sphere_search_recursive(const OctreeNode* node,
                                         const std::array<real_t,3>& center,
                                         real_t radius,
                                         std::unordered_set<index_t>& results) const {
  if (!node || !search::aabb_sphere_overlap(node->bounds, center, radius)) {
    return;
  }

  if (node->is_leaf) {
    // Check cells in this node
    for (index_t cell_id : node->cell_indices) {
      if (search::aabb_sphere_overlap(cell_aabbs_[cell_id], center, radius)) {
        results.insert(cell_id);
      }
    }
  } else {
    // Recurse to overlapping children
    for (int octant = 0; octant < 8; ++octant) {
      if (node->children[octant]) {
        sphere_search_recursive(node->children[octant].get(),
                              center, radius, results);
      }
    }
  }
}

// ---- Helper methods ----

const OctreeAccel::OctreeNode* OctreeAccel::find_leaf(
    const std::array<real_t,3>& point) const {

  const OctreeNode* node = root_.get();

  while (node && !node->is_leaf) {
    int octant = node->get_octant(point);
    node = node->children[octant].get();
  }

  return node;
}

real_t OctreeAccel::distance_to_node(const OctreeNode* node,
                                    const std::array<real_t,3>& point) const {
  if (!node) return std::numeric_limits<real_t>::max();
  return search::point_aabb_distance(point, node->bounds);
}

bool OctreeAccel::ray_intersects_node(const OctreeNode* node,
                                      const search::Ray& ray,
                                      real_t& t_near,
                                      real_t& t_far) const {
  if (!node) return false;
  return search::ray_aabb_intersect(ray, node->bounds, t_near, t_far);
}

size_t OctreeAccel::count_nodes(const OctreeNode* node) const {
  if (!node) return 0;
  if (node->is_leaf) return 1;

  size_t count = 1;
  for (int i = 0; i < 8; ++i) {
    count += count_nodes(node->children[i].get());
  }
  return count;
}

int OctreeAccel::get_max_depth(const OctreeNode* node) const {
  if (!node || node->is_leaf) return node ? node->depth : 0;

  int max_depth = node->depth;
  for (int i = 0; i < 8; ++i) {
    if (node->children[i]) {
      max_depth = std::max(max_depth, get_max_depth(node->children[i].get()));
    }
  }
  return max_depth;
}

size_t OctreeAccel::compute_memory_usage() const {
  size_t total = 0;

  // Node memory
  total += sizeof(OctreeNode) * count_nodes(root_.get());

  // Vertex and cell data
  total += sizeof(std::array<real_t,3>) * vertex_coords_.size();
  total += sizeof(search::AABB) * cell_aabbs_.size();
  total += sizeof(BoundaryTriangle) * boundary_triangles_.size();

  // Entity lists in leaves
  std::vector<const OctreeNode*> leaves;
  collect_leaves(root_.get(), leaves);

  for (const OctreeNode* leaf : leaves) {
    total += sizeof(index_t) * (leaf->cell_indices.size() +
                                leaf->vertex_indices.size());
  }

  return total;
}

void OctreeAccel::collect_leaves(const OctreeNode* node,
                                std::vector<const OctreeNode*>& leaves) const {
  if (!node) return;

  if (node->is_leaf) {
    leaves.push_back(node);
  } else {
    for (int i = 0; i < 8; ++i) {
      collect_leaves(node->children[i].get(), leaves);
    }
  }
}

std::vector<int> OctreeAccel::get_overlapping_children(const OctreeNode* node,
                                                       const search::AABB& box) const {
  std::vector<int> overlapping;

  for (int octant = 0; octant < 8; ++octant) {
    if (node->children[octant]) {
      auto child_bounds = node->get_child_bounds(octant);
      if (child_bounds.overlaps(box)) {
        overlapping.push_back(octant);
      }
    }
  }

  return overlapping;
}

} // namespace svmp