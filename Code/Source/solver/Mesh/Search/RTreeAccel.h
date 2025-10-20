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

#ifndef SVMP_RTREE_ACCEL_H
#define SVMP_RTREE_ACCEL_H

#include "SearchAccel.h"
#include "SearchPrimitives.h"
#include <vector>
#include <memory>
#include <queue>
#include <array>
#include <algorithm>
#include <unordered_set>
#include <limits>

namespace svmp {

/**
 * @brief R-Tree spatial acceleration structure
 *
 * A balanced tree data structure for spatial indexing that groups
 * nearby objects and represents them with their minimum bounding
 * rectangle (MBR) at higher levels. Particularly efficient for:
 * - Dynamic insertions and deletions
 * - Range queries and spatial searches
 * - Disk-based spatial databases
 * - Geographic information systems (GIS)
 *
 * This implementation uses the R*-tree variant with improved
 * node splitting and reinsertion strategies for better query performance.
 */
class RTreeAccel : public IAccel {
public:
  RTreeAccel() = default;
  virtual ~RTreeAccel() = default;

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

  // ---- R-Tree specific methods ----

  /**
   * @brief Insert a new entity into the R-Tree
   *
   * Supports dynamic insertion after initial build.
   */
  void insert(index_t entity_id, const search::AABB& bounds);

  /**
   * @brief Remove an entity from the R-Tree
   *
   * Supports dynamic deletion with tree rebalancing.
   */
  void remove(index_t entity_id, const search::AABB& bounds);

  /**
   * @brief Bulk load entities using Sort-Tile-Recursive (STR) algorithm
   *
   * More efficient than individual insertions for initial build.
   */
  void bulk_load(const std::vector<std::pair<index_t, search::AABB>>& entries);

private:
  // ---- R-Tree node structure ----

  struct RTreeNode {
    // Node bounds (Minimum Bounding Rectangle - MBR)
    search::AABB mbr;

    // Node type
    bool is_leaf = false;
    int level = 0;  // Leaf level is 0

    // Parent pointer for tree traversal
    RTreeNode* parent = nullptr;

    // For leaf nodes: entity data
    struct Entry {
      index_t id;
      search::AABB bounds;

      Entry() : id(-1) {}
      Entry(index_t i, const search::AABB& b) : id(i), bounds(b) {}
    };
    std::vector<Entry> entries;  // Only used in leaf nodes

    // For internal nodes: child pointers
    std::vector<std::unique_ptr<RTreeNode>> children;

    // Node capacity constraints
    static constexpr int MIN_ENTRIES = 2;  // Minimum entries (m)
    static constexpr int MAX_ENTRIES = 6;  // Maximum entries (M)

    RTreeNode(bool leaf = false, int lvl = 0)
        : is_leaf(leaf), level(lvl) {
      if (is_leaf) {
        entries.reserve(MAX_ENTRIES + 1);  // +1 for overflow before split
      } else {
        children.reserve(MAX_ENTRIES + 1);
      }
    }

    // Check if node is full
    bool is_full() const {
      return is_leaf ? entries.size() >= MAX_ENTRIES
                     : children.size() >= MAX_ENTRIES;
    }

    // Check if node is underfull
    bool is_underfull() const {
      return is_leaf ? entries.size() < MIN_ENTRIES
                     : children.size() < MIN_ENTRIES;
    }

    // Update MBR from entries or children
    void update_mbr() {
      bool first = true;
      if (is_leaf) {
        for (const auto& entry : entries) {
          if (first) {
            mbr = entry.bounds;
            first = false;
          } else {
            mbr.expand(entry.bounds);
          }
        }
      } else {
        for (const auto& child : children) {
          if (first) {
            mbr = child->mbr;
            first = false;
          } else {
            mbr.expand(child->mbr);
          }
        }
      }
    }

    // Get number of entries
    size_t size() const {
      return is_leaf ? entries.size() : children.size();
    }
  };

  // Split result structure
  struct SplitResult {
    std::unique_ptr<RTreeNode> first;
    std::unique_ptr<RTreeNode> second;
  };

  // Nearest neighbor search entry
  struct NNEntry {
    RTreeNode* node;
    real_t min_dist;

    bool operator>(const NNEntry& other) const {
      return min_dist > other.min_dist;
    }
  };

  // ---- Tree data ----

  std::unique_ptr<RTreeNode> root_;
  int tree_height_ = 0;

  // Cached mesh data
  std::vector<std::array<real_t,3>> vertex_coords_;
  std::vector<search::AABB> cell_aabbs_;
  std::vector<index_t> cell_indices_;

  // For vertices (nearest neighbor)
  std::vector<index_t> vertex_indices_;
  std::unique_ptr<RTreeNode> vertex_root_;  // Separate tree for vertices

  // For boundary triangles (ray intersection)
  struct BoundaryTriangle {
    std::array<std::array<real_t,3>,3> vertices;
    std::array<real_t,3> normal;
    index_t face_id;
  };
  std::vector<BoundaryTriangle> boundary_triangles_;

  // State
  bool is_built_ = false;
  Configuration built_cfg_ = Configuration::Reference;
  mutable SearchStats stats_;

  // R*-tree parameters
  int reinsert_count_ = 0;
  static constexpr int MAX_REINSERT_LEVEL = 1;
  static constexpr real_t REINSERT_FRACTION = 0.3;

  // ---- Helper methods for tree construction ----

  /**
   * @brief Choose leaf node for insertion
   */
  RTreeNode* choose_leaf(RTreeNode* node, const search::AABB& bounds, int target_level);

  /**
   * @brief Choose subtree for insertion (R*-tree algorithm)
   */
  RTreeNode* choose_subtree(RTreeNode* node, const search::AABB& bounds);

  /**
   * @brief Insert entry into leaf node
   */
  void insert_entry(RTreeNode* leaf, index_t id, const search::AABB& bounds);

  /**
   * @brief Insert child into internal node
   */
  void insert_child(RTreeNode* node, std::unique_ptr<RTreeNode> child);

  /**
   * @brief Adjust tree after insertion
   */
  void adjust_tree(RTreeNode* node, std::unique_ptr<RTreeNode> split_node = nullptr);

  /**
   * @brief Split node using quadratic split algorithm
   */
  SplitResult quadratic_split(RTreeNode* node);

  /**
   * @brief Split node using linear split algorithm (faster, lower quality)
   */
  SplitResult linear_split(RTreeNode* node);

  /**
   * @brief Split node using R*-tree split algorithm
   */
  SplitResult rstar_split(RTreeNode* node);

  /**
   * @brief Pick seeds for quadratic split
   */
  std::pair<int, int> pick_seeds_quadratic(RTreeNode* node);

  /**
   * @brief Pick next entry for quadratic split
   */
  int pick_next_quadratic(RTreeNode* node,
                          const std::vector<bool>& assigned,
                          const search::AABB& mbr1,
                          const search::AABB& mbr2);

  /**
   * @brief Compute area enlargement needed
   */
  real_t compute_enlargement(const search::AABB& mbr, const search::AABB& new_mbr) const;

  /**
   * @brief Compute overlap between MBRs
   */
  real_t compute_overlap(const search::AABB& mbr1, const search::AABB& mbr2) const;

  /**
   * @brief R*-tree forced reinsert
   */
  void reinsert(RTreeNode* node, int level);

  /**
   * @brief Sort-Tile-Recursive bulk loading
   */
  std::unique_ptr<RTreeNode> str_build(
      std::vector<std::pair<index_t, search::AABB>>& entries,
      int start, int end, int level);

  /**
   * @brief Compute optimal node capacity for STR
   */
  int compute_str_node_capacity(int n_entries) const;

  // ---- Helper methods for deletion ----

  /**
   * @brief Find leaf containing entry
   */
  RTreeNode* find_leaf(RTreeNode* node, index_t id, const search::AABB& bounds);

  /**
   * @brief Condense tree after deletion
   */
  void condense_tree(RTreeNode* leaf);

  /**
   * @brief Reinsert orphaned entries
   */
  void reinsert_orphans(const std::vector<std::unique_ptr<RTreeNode>>& orphans);

  // ---- Helper methods for queries ----

  /**
   * @brief Search for entries overlapping with query box
   */
  void search_recursive(RTreeNode* node,
                       const search::AABB& query,
                       std::vector<index_t>& results) const;

  /**
   * @brief Point location in R-Tree
   */
  void locate_point_recursive(RTreeNode* node,
                             const MeshBase& mesh,
                             const std::array<real_t,3>& point,
                             PointLocateResult& result) const;

  /**
   * @brief Nearest neighbor search with branch-and-bound
   */
  void nearest_neighbor_recursive(RTreeNode* node,
                                 const std::array<real_t,3>& point,
                                 index_t& best_idx,
                                 real_t& best_dist_sq) const;

  /**
   * @brief k-nearest neighbors with priority queue
   */
  void k_nearest_search(RTreeNode* root,
                       const std::array<real_t,3>& point,
                       size_t k,
                       std::priority_queue<std::pair<real_t, index_t>>& max_heap) const;

  /**
   * @brief Find vertices within radius
   */
  void radius_search_recursive(RTreeNode* node,
                              const std::array<real_t,3>& point,
                              real_t radius_sq,
                              std::vector<index_t>& results) const;

  /**
   * @brief Ray traversal through R-Tree
   */
  void ray_traverse_recursive(RTreeNode* node,
                             const search::Ray& ray,
                             std::vector<RayIntersectResult>& results) const;

  /**
   * @brief Test ray against triangles in node
   */
  bool ray_node_intersection(RTreeNode* node,
                            const search::Ray& ray,
                            RayIntersectResult& result) const;

  /**
   * @brief Find cells overlapping sphere
   */
  void sphere_search_recursive(RTreeNode* node,
                              const std::array<real_t,3>& center,
                              real_t radius,
                              std::unordered_set<index_t>& results) const;

  // ---- Performance helpers ----

  /**
   * @brief Minimum distance from point to MBR
   */
  real_t min_distance_to_mbr(const search::AABB& mbr,
                            const std::array<real_t,3>& point) const;

  /**
   * @brief Maximum distance from point to MBR
   */
  real_t max_distance_to_mbr(const search::AABB& mbr,
                            const std::array<real_t,3>& point) const;

  /**
   * @brief Check if ray intersects MBR
   */
  bool ray_intersects_mbr(const search::Ray& ray,
                         const search::AABB& mbr,
                         real_t& t_near,
                         real_t& t_far) const;

  /**
   * @brief Test ray-triangle intersection
   */
  bool ray_triangle_intersection(const search::Ray& ray,
                                const BoundaryTriangle& tri,
                                real_t& t,
                                std::array<real_t,3>& hit_point,
                                std::array<real_t,2>& bary) const;

  /**
   * @brief Count nodes in subtree
   */
  size_t count_nodes(RTreeNode* node) const;

  /**
   * @brief Get tree height
   */
  int get_tree_height(RTreeNode* node) const;

  /**
   * @brief Compute memory usage
   */
  size_t compute_memory_usage() const;

  /**
   * @brief Validate tree structure (for debugging)
   */
  bool validate_tree(RTreeNode* node = nullptr) const;

  /**
   * @brief Get tree utilization statistics
   */
  void compute_utilization(RTreeNode* node,
                          real_t& avg_fullness,
                          real_t& avg_overlap) const;
};

} // namespace svmp

#endif // SVMP_RTREE_ACCEL_H