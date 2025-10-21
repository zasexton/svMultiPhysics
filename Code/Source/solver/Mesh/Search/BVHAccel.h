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

#ifndef SVMP_BVH_ACCEL_H
#define SVMP_BVH_ACCEL_H

#include "SearchAccel.h"
#include "SearchPrimitives.h"
#include <vector>
#include <memory>
#include <queue>
#include <array>
#include <algorithm>
#include <unordered_set>

namespace svmp {

/**
 * @brief Bounding Volume Hierarchy acceleration structure
 *
 * A binary tree where each node contains a bounding volume (AABB)
 * that encloses all its children. Particularly efficient for:
 * - Ray tracing and intersection queries
 * - Non-uniform spatial distributions
 * - Dynamic scene updates (through refitting)
 * - Memory-efficient representation
 *
 * This implementation uses Surface Area Heuristic (SAH) for
 * optimal tree construction, balancing tree quality and build time.
 */
class BVHAccel : public IAccel {
public:
  BVHAccel() = default;
  virtual ~BVHAccel() = default;

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

  // ---- BVH-specific methods ----

  /**
   * @brief Refit the BVH after geometry changes
   *
   * Updates bounding boxes without rebuilding the tree structure.
   * Much faster than rebuild but may result in suboptimal tree.
   */
  void refit(const MeshBase& mesh);

  /**
   * @brief Get the Surface Area Heuristic cost of the tree
   */
  real_t compute_sah_cost() const;

private:
  // ---- BVH node structure ----

  struct BVHNode {
    // Node bounds
    search::AABB bounds;

    // Node type and data
    bool is_leaf = false;

    // Leaf data - primitive indices
    std::vector<index_t> primitive_indices;  // Cell or triangle indices

    // Internal node data - children
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;

    // Additional metadata for optimization
    int depth = 0;
    real_t area = 0;  // Surface area of bounds (cached for SAH)

    BVHNode() = default;
    BVHNode(const search::AABB& b, int d) : bounds(b), depth(d) {
      auto extents = bounds.extents();
      area = 2.0 * (extents[0]*extents[1] + extents[1]*extents[2] + extents[2]*extents[0]);
    }

    // Update surface area after bounds change
    void update_area() {
      auto extents = bounds.extents();
      area = 2.0 * (extents[0]*extents[1] + extents[1]*extents[2] + extents[2]*extents[0]);
    }
  };

  // Build primitive info
  struct PrimitiveInfo {
    index_t primitive_index;
    search::AABB bounds;
    std::array<real_t,3> centroid;

    PrimitiveInfo() = default;
    PrimitiveInfo(index_t idx, const search::AABB& b)
        : primitive_index(idx), bounds(b), centroid(b.center()) {}
  };

  // SAH bucket for binning
  struct SAHBucket {
    int count = 0;
    search::AABB bounds;
  };

  // Traversal stack entry
  struct StackEntry {
    const BVHNode* node;
    real_t t_min;
    real_t t_max;
  };

  // Priority queue entry for nearest neighbor
  struct NearestEntry {
    const BVHNode* node;
    real_t distance;

    bool operator>(const NearestEntry& other) const {
      return distance > other.distance;
    }
  };

  // ---- Tree data ----

  std::unique_ptr<BVHNode> root_;

  // Cached mesh data
  std::vector<std::array<real_t,3>> vertex_coords_;
  std::vector<search::AABB> primitive_aabbs_;

  // For cells
  std::vector<index_t> cell_indices_;

  // For boundary triangles (ray intersection)
  struct BoundaryTriangle {
    std::array<std::array<real_t,3>,3> vertices;
    std::array<real_t,3> normal;
    index_t face_id;
  };
  std::vector<BoundaryTriangle> boundary_triangles_;

  // For vertices (nearest neighbor)
  std::vector<index_t> vertex_indices_;
  std::vector<std::vector<index_t>> vertex_to_cells_;

  // State
  bool is_built_ = false;
  Configuration built_cfg_ = Configuration::Reference;
  mutable SearchStats stats_;

  // Build parameters
  int max_depth_ = 40;
  int min_primitives_per_leaf_ = 1;
  int max_primitives_per_leaf_ = 16;

  // SAH parameters
  real_t traversal_cost_ = 1.0;
  real_t intersection_cost_ = 1.0;
  int n_sah_buckets_ = 12;  // Number of buckets for SAH binning

  // Build method
  enum class BuildMethod {
    SAH,          // Surface Area Heuristic (best quality)
    HLBVH,        // Hierarchical Linear BVH (faster build)
    Middle,       // Simple middle split
    EqualCounts   // Equal primitive counts
  };
  BuildMethod build_method_ = BuildMethod::SAH;

  // ---- Helper methods for tree construction ----

  /**
   * @brief Build BVH recursively using chosen heuristic
   */
  std::unique_ptr<BVHNode> build_recursive(
      std::vector<PrimitiveInfo>& primitives,
      int start, int end,
      int depth);

  /**
   * @brief Find best split using Surface Area Heuristic
   */
  int find_sah_split(const std::vector<PrimitiveInfo>& primitives,
                     int start, int end,
                     int& split_axis,
                     real_t& split_cost);

  /**
   * @brief Partition primitives based on SAH split
   */
  int partition_sah(std::vector<PrimitiveInfo>& primitives,
                   int start, int end,
                   int axis,
                   real_t split_pos);

  /**
   * @brief Create leaf node
   */
  std::unique_ptr<BVHNode> create_leaf(
      const std::vector<PrimitiveInfo>& primitives,
      int start, int end,
      int depth);

  /**
   * @brief Compute bounds for primitive range
   */
  search::AABB compute_bounds(const std::vector<PrimitiveInfo>& primitives,
                             int start, int end) const;

  /**
   * @brief Compute centroid bounds for primitive range
   */
  search::AABB compute_centroid_bounds(const std::vector<PrimitiveInfo>& primitives,
                                      int start, int end) const;

  // ---- Helper methods for queries ----

  /**
   * @brief Traverse BVH for point location
   */
  void locate_point_recursive(const BVHNode* node,
                            const MeshBase& mesh,
                            const std::array<real_t,3>& point,
                            PointLocateResult& result) const;

  /**
   * @brief Find nearest vertex using priority queue
   */
  void nearest_vertex_recursive(const BVHNode* node,
                              const std::array<real_t,3>& point,
                              index_t& best_idx,
                              real_t& best_dist_sq) const;

  /**
   * @brief k-nearest neighbors with priority queue
   */
  void k_nearest_recursive(const BVHNode* node,
                         const std::array<real_t,3>& point,
                         size_t k,
                         std::priority_queue<std::pair<real_t, index_t>>& max_heap) const;

  /**
   * @brief Find vertices within radius
   */
  void radius_search_recursive(const BVHNode* node,
                             const std::array<real_t,3>& point,
                             real_t radius_sq,
                             std::vector<index_t>& results) const;

  /**
   * @brief Intersect ray with BVH using stack-based traversal
   */
  bool intersect_ray_node(const BVHNode* node,
                         const search::Ray& ray,
                         RayIntersectResult& result,
                         real_t& t_min) const;

  /**
   * @brief Find all ray intersections
   */
  void intersect_ray_all_recursive(const BVHNode* node,
                                  const search::Ray& ray,
                                  std::vector<RayIntersectResult>& results) const;

  /**
   * @brief Find primitives overlapping box
   */
  void box_search_recursive(const BVHNode* node,
                          const search::AABB& box,
                          std::unordered_set<index_t>& results) const;

  /**
   * @brief Find primitives overlapping sphere
   */
  void sphere_search_recursive(const BVHNode* node,
                             const std::array<real_t,3>& center,
                             real_t radius,
                             std::unordered_set<index_t>& results) const;

  // ---- Performance helpers ----

  /**
   * @brief Distance from point to node bounds
   */
  real_t distance_to_node(const BVHNode* node,
                        const std::array<real_t,3>& point) const;

  /**
   * @brief Check ray-AABB intersection with near/far distances
   */
  bool ray_intersects_box(const search::Ray& ray,
                        const search::AABB& box,
                        real_t& t_near,
                        real_t& t_far) const;

  /**
   * @brief Test ray against triangle
   */
  bool ray_triangle_intersection(const search::Ray& ray,
                               const BoundaryTriangle& tri,
                               real_t& t,
                               std::array<real_t,3>& hit_point,
                               std::array<real_t,2>& bary) const;

  /**
   * @brief Refit node bounds recursively
   */
  void refit_recursive(BVHNode* node);

  /**
   * @brief Compute SAH cost recursively
   */
  real_t compute_sah_cost_recursive(const BVHNode* node) const;

  /**
   * @brief Count nodes in subtree
   */
  size_t count_nodes(const BVHNode* node) const;

  /**
   * @brief Get maximum tree depth
   */
  int get_max_depth(const BVHNode* node) const;

  /**
   * @brief Compute memory usage
   */
  size_t compute_memory_usage() const;

  /**
   * @brief Flatten BVH for cache-efficient traversal (optional optimization)
   */
  void flatten_tree();
};

} // namespace svmp

#endif // SVMP_BVH_ACCEL_H
