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

#ifndef SVMP_KDTREE_ACCEL_H
#define SVMP_KDTREE_ACCEL_H

#include "SearchAccel.h"
#include "SearchPrimitives.h"
#include <vector>
#include <memory>
#include <queue>
#include <limits>

namespace svmp {

/**
 * @brief KD-Tree spatial acceleration structure
 *
 * A k-dimensional tree (k=3 for 3D space) that partitions space
 * using axis-aligned splitting planes. Particularly efficient for
 * nearest neighbor queries on vertices.
 */
class KDTreeAccel : public IAccel {
public:
  KDTreeAccel() = default;
  virtual ~KDTreeAccel() = default;

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
  // ---- KD-Tree node structure ----

  struct KDNode {
    // Leaf node data
    bool is_leaf = false;
    std::vector<index_t> point_indices;  // Vertex indices in this leaf

    // Internal node data
    int split_axis = -1;                 // 0=x, 1=y, 2=z
    real_t split_value = 0;              // Split plane position
    std::unique_ptr<KDNode> left;        // Left child (values <= split)
    std::unique_ptr<KDNode> right;       // Right child (values > split)

    // Node bounds
    search::AABB bounds;

    KDNode() = default;
  };

  // ---- Tree data ----

  std::unique_ptr<KDNode> root_;           // Root node of the tree
  std::vector<std::array<real_t,3>> vertices_;  // Vertex positions
  std::vector<index_t> vertex_indices_;    // Original vertex indices

  // For cell queries (secondary structure)
  std::vector<search::AABB> cell_aabbs_;   // AABBs for cells
  std::vector<std::vector<index_t>> vertex_to_cells_;  // Vertex to cell mapping

  // State
  bool is_built_ = false;
  Configuration built_cfg_ = Configuration::Reference;
  SearchStats stats_;

  // Build parameters
  int max_depth_ = 20;
  int min_points_per_leaf_ = 10;

  // ---- Helper methods for tree construction ----

  /**
   * @brief Recursively build KD-tree nodes
   */
  std::unique_ptr<KDNode> build_node(
      std::vector<index_t>& indices,
      int start, int end,
      int depth);

  /**
   * @brief Choose split axis using cycling or spread-based heuristic
   */
  int choose_split_axis(const std::vector<index_t>& indices,
                       int start, int end,
                       int depth);

  /**
   * @brief Partition points around median along given axis
   */
  int partition_points(std::vector<index_t>& indices,
                      int start, int end,
                      int axis,
                      real_t& split_value);

  /**
   * @brief Compute bounding box for a range of points
   */
  search::AABB compute_bounds(const std::vector<index_t>& indices,
                             int start, int end) const;

  // ---- Helper methods for queries ----

  /**
   * @brief Recursively search for nearest neighbor
   */
  void nearest_neighbor_recursive(
      const KDNode* node,
      const std::array<real_t,3>& point,
      index_t& best_idx,
      real_t& best_dist_sq) const;

  /**
   * @brief Recursively search for k nearest neighbors
   */
  void k_nearest_recursive(
      const KDNode* node,
      const std::array<real_t,3>& point,
      size_t k,
      std::priority_queue<std::pair<real_t, index_t>>& max_heap) const;

  /**
   * @brief Recursively find vertices within radius
   */
  void radius_search_recursive(
      const KDNode* node,
      const std::array<real_t,3>& point,
      real_t radius_sq,
      std::vector<index_t>& results) const;

  /**
   * @brief Find vertices in axis-aligned box
   */
  void box_search_recursive(
      const KDNode* node,
      const search::AABB& box,
      std::vector<index_t>& results) const;

  /**
   * @brief Distance from point to node's bounding box
   */
  real_t distance_to_node_bounds(const KDNode* node,
                                const std::array<real_t,3>& point) const;

  /**
   * @brief Build vertex-to-cells mapping for cell queries
   */
  void build_vertex_to_cells(const MeshBase& mesh);

  /**
   * @brief Get cells associated with nearest vertices
   */
  std::vector<index_t> get_cells_from_vertices(
      const std::vector<index_t>& vertex_indices) const;

  /**
   * @brief Linear search fallback for point location
   */
  PointLocateResult locate_point_linear(
      const MeshBase& mesh,
      const std::array<real_t,3>& point) const;

  // ---- Performance helpers ----

  /**
   * @brief Balance factor for subtree
   */
  real_t compute_balance_factor(const KDNode* node) const;

  /**
   * @brief Count nodes in subtree
   */
  size_t count_nodes(const KDNode* node) const;

  /**
   * @brief Compute tree depth
   */
  int compute_tree_depth(const KDNode* node) const;
};

} // namespace svmp

#endif // SVMP_KDTREE_ACCEL_H