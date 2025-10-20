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

#ifndef SVMP_OCTREE_ACCEL_H
#define SVMP_OCTREE_ACCEL_H

#include "SearchAccel.h"
#include "SearchPrimitives.h"
#include <vector>
#include <memory>
#include <queue>
#include <array>
#include <unordered_set>

namespace svmp {

/**
 * @brief Octree spatial acceleration structure
 *
 * A hierarchical tree structure that recursively subdivides 3D space
 * into eight octants. Each node either contains entities (leaf) or
 * has eight children representing spatial subdivisions.
 *
 * Particularly efficient for:
 * - Non-uniform spatial distributions
 * - Hierarchical culling
 * - Ray tracing
 * - Level-of-detail queries
 */
class OctreeAccel : public IAccel {
public:
  OctreeAccel() = default;
  virtual ~OctreeAccel() = default;

  // ---- Building ----

  void build(const MeshBase& mesh,
            Configuration cfg,
            const MeshSearch::SearchConfig& config) override;

  void clear() override;

  bool is_built() const override {
    return is_built_ && root_ != nullptr;
  }

  Configuration built_config() const override {
    return built_cfg_;
  }

  // ---- Point location ----

  PointLocateResult locate_point(const MeshBase& mesh,
                                const std::array<real_t,3>& point,
                                index_t hint_cell = -1) const override;

  std::vector<PointLocateResult> locate_points(
      const MeshBase& mesh,
      const std::vector<std::array<real_t,3>>& points) const override;

  // ---- Nearest neighbor ----

  std::pair<index_t, real_t> nearest_vertex(
      const MeshBase& mesh,
      const std::array<real_t,3>& point) const override;

  std::vector<std::pair<index_t, real_t>> k_nearest_vertices(
      const MeshBase& mesh,
      const std::array<real_t,3>& point,
      size_t k) const override;

  std::vector<index_t> vertices_in_radius(
      const MeshBase& mesh,
      const std::array<real_t,3>& point,
      real_t radius) const override;

  // ---- Ray intersection ----

  RayIntersectResult intersect_ray(
      const MeshBase& mesh,
      const std::array<real_t,3>& origin,
      const std::array<real_t,3>& direction,
      real_t max_distance = 1e300) const override;

  std::vector<RayIntersectResult> intersect_ray_all(
      const MeshBase& mesh,
      const std::array<real_t,3>& origin,
      const std::array<real_t,3>& direction,
      real_t max_distance = 1e300) const override;

  // ---- Region queries ----

  std::vector<index_t> cells_in_box(
      const MeshBase& mesh,
      const std::array<real_t,3>& box_min,
      const std::array<real_t,3>& box_max) const override;

  std::vector<index_t> cells_in_sphere(
      const MeshBase& mesh,
      const std::array<real_t,3>& center,
      real_t radius) const override;

  // ---- Statistics ----

  SearchStats get_stats() const override {
    return stats_;
  }

private:
  // ---- Octree node structure ----

  struct OctreeNode {
    // Node bounds
    search::AABB bounds;
    std::array<real_t,3> center;
    real_t half_size;

    // Node data
    bool is_leaf = false;
    int depth = 0;

    // Leaf data - store entity indices
    std::vector<index_t> cell_indices;     // Cell indices in this node
    std::vector<index_t> vertex_indices;   // Vertex indices in this node

    // Internal node data - children
    std::array<std::unique_ptr<OctreeNode>, 8> children;

    // Constructor
    OctreeNode(const search::AABB& b, int d)
        : bounds(b), depth(d) {
      center = bounds.center();
      auto extents = bounds.extents();
      half_size = std::max({extents[0], extents[1], extents[2]}) * 0.5;
    }

    // Get child octant for a point (0-7)
    int get_octant(const std::array<real_t,3>& point) const {
      int octant = 0;
      if (point[0] > center[0]) octant |= 1;
      if (point[1] > center[1]) octant |= 2;
      if (point[2] > center[2]) octant |= 4;
      return octant;
    }

    // Get bounds for a child octant
    search::AABB get_child_bounds(int octant) const {
      std::array<real_t,3> child_min = center;
      std::array<real_t,3> child_max = center;

      for (int i = 0; i < 3; ++i) {
        if (octant & (1 << i)) {
          child_max[i] = bounds.max[i];
        } else {
          child_min[i] = bounds.min[i];
          child_max[i] = center[i];
        }

        if (!(octant & (1 << i))) {
          child_min[i] = bounds.min[i];
        } else {
          child_min[i] = center[i];
        }
      }

      return search::AABB(child_min, child_max);
    }
  };

  // Priority queue entry for traversal
  struct TraversalEntry {
    const OctreeNode* node;
    real_t priority;  // Distance or other metric

    bool operator<(const TraversalEntry& other) const {
      return priority > other.priority;  // Min heap
    }
  };

  // ---- Tree data ----

  std::unique_ptr<OctreeNode> root_;

  // Cached mesh data
  std::vector<std::array<real_t,3>> vertex_coords_;
  std::vector<search::AABB> cell_aabbs_;

  // For boundary triangles (ray intersection)
  struct BoundaryTriangle {
    std::array<std::array<real_t,3>,3> vertices;
    index_t face_id;
  };
  std::vector<BoundaryTriangle> boundary_triangles_;

  // State
  bool is_built_ = false;
  Configuration built_cfg_ = Configuration::Reference;
  SearchStats stats_;

  // Build parameters
  int max_depth_ = 10;
  int min_entities_per_leaf_ = 10;
  int max_entities_per_leaf_ = 100;

  // ---- Helper methods for tree construction ----

  /**
   * @brief Recursively build octree nodes
   */
  void build_node(OctreeNode* node,
                 const std::vector<index_t>& cell_indices,
                 const std::vector<index_t>& vertex_indices);

  /**
   * @brief Determine if node should be subdivided
   */
  bool should_subdivide(const OctreeNode* node,
                       size_t n_cells,
                       size_t n_vertices) const;

  /**
   * @brief Distribute entities to child octants
   */
  void distribute_entities(const OctreeNode* parent,
                         const std::vector<index_t>& cell_indices,
                         const std::vector<index_t>& vertex_indices,
                         std::array<std::vector<index_t>, 8>& child_cells,
                         std::array<std::vector<index_t>, 8>& child_vertices);

  // ---- Helper methods for queries ----

  /**
   * @brief Find leaf node containing point
   */
  const OctreeNode* find_leaf(const std::array<real_t,3>& point) const;

  /**
   * @brief Recursively search for cells containing point
   */
  void locate_point_recursive(const OctreeNode* node,
                             const MeshBase& mesh,
                             const std::array<real_t,3>& point,
                             PointLocateResult& result) const;

  /**
   * @brief Recursively find nearest vertex
   */
  void nearest_vertex_recursive(const OctreeNode* node,
                               const std::array<real_t,3>& point,
                               index_t& best_idx,
                               real_t& best_dist_sq) const;

  /**
   * @brief Priority-based k-nearest search
   */
  void k_nearest_search(const std::array<real_t,3>& point,
                       size_t k,
                       std::priority_queue<std::pair<real_t, index_t>>& max_heap) const;

  /**
   * @brief Find vertices within radius
   */
  void radius_search_recursive(const OctreeNode* node,
                              const std::array<real_t,3>& point,
                              real_t radius_sq,
                              std::vector<index_t>& results) const;

  /**
   * @brief Ray traversal through octree
   */
  void ray_traverse(const OctreeNode* node,
                   const search::Ray& ray,
                   std::vector<const OctreeNode*>& leaves) const;

  /**
   * @brief Test ray against triangles in node
   */
  bool ray_node_intersection(const OctreeNode* node,
                            const search::Ray& ray,
                            RayIntersectResult& result) const;

  /**
   * @brief Find cells overlapping box
   */
  void box_search_recursive(const OctreeNode* node,
                           const search::AABB& box,
                           std::unordered_set<index_t>& results) const;

  /**
   * @brief Find cells overlapping sphere
   */
  void sphere_search_recursive(const OctreeNode* node,
                              const std::array<real_t,3>& center,
                              real_t radius,
                              std::unordered_set<index_t>& results) const;

  // ---- Performance helpers ----

  /**
   * @brief Distance from point to node bounds
   */
  real_t distance_to_node(const OctreeNode* node,
                         const std::array<real_t,3>& point) const;

  /**
   * @brief Check if ray intersects node bounds
   */
  bool ray_intersects_node(const OctreeNode* node,
                          const search::Ray& ray,
                          real_t& t_near,
                          real_t& t_far) const;

  /**
   * @brief Count nodes in subtree
   */
  size_t count_nodes(const OctreeNode* node) const;

  /**
   * @brief Get maximum tree depth
   */
  int get_max_depth(const OctreeNode* node) const;

  /**
   * @brief Compute memory usage
   */
  size_t compute_memory_usage() const;

  /**
   * @brief Collect all leaf nodes
   */
  void collect_leaves(const OctreeNode* node,
                     std::vector<const OctreeNode*>& leaves) const;

  /**
   * @brief Get child nodes that overlap with AABB
   */
  std::vector<int> get_overlapping_children(const OctreeNode* node,
                                           const search::AABB& box) const;
};

} // namespace svmp

#endif // SVMP_OCTREE_ACCEL_H