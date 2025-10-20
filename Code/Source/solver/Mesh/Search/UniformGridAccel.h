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

#ifndef SVMP_UNIFORM_GRID_ACCEL_H
#define SVMP_UNIFORM_GRID_ACCEL_H

#include "SearchAccel.h"
#include "SearchPrimitives.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace svmp {

/**
 * @brief Uniform grid spatial acceleration structure
 *
 * Divides space into a regular grid of cells. Each grid cell
 * stores a list of mesh entities that overlap with it.
 */
class UniformGridAccel : public IAccel {
public:
  UniformGridAccel() = default;
  virtual ~UniformGridAccel() = default;

  // ---- Building ----

  void build(const MeshBase& mesh,
            Configuration cfg,
            const MeshSearch::SearchConfig& config) override;

  void clear() override;

  bool is_built() const override {
    return is_built_;
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
  // ---- Grid structure ----

  struct GridCell {
    std::vector<index_t> entities;  // Cell/vertex/face indices in this grid cell
  };

  // Grid data
  std::array<int,3> resolution_;     // Grid resolution in each dimension
  search::AABB bounds_;               // Overall bounding box
  std::array<real_t,3> cell_size_;   // Size of each grid cell
  std::vector<GridCell> grid_cells_; // Flattened grid array

  // Cell AABBs (cached for efficiency)
  std::vector<search::AABB> cell_aabbs_;

  // Vertex positions (for nearest neighbor queries)
  std::vector<std::array<real_t,3>> vertex_coords_;

  // Boundary face triangles (for ray intersection)
  struct BoundaryTriangle {
    std::array<std::array<real_t,3>,3> vertices;
    index_t face_id;
  };
  std::vector<BoundaryTriangle> boundary_triangles_;

  // State
  bool is_built_ = false;
  Configuration built_cfg_ = Configuration::Reference;
  SearchStats stats_;

  // ---- Helper methods ----

  /**
   * @brief Convert 3D grid indices to linear index
   */
  size_t grid_index(int i, int j, int k) const {
    if (i < 0 || i >= resolution_[0] ||
        j < 0 || j >= resolution_[1] ||
        k < 0 || k >= resolution_[2]) {
      return SIZE_MAX;  // Invalid index
    }
    return k * resolution_[0] * resolution_[1] + j * resolution_[0] + i;
  }

  /**
   * @brief Get grid indices for a point
   */
  std::array<int,3> point_to_grid(const std::array<real_t,3>& point) const;

  /**
   * @brief Get grid cells overlapping with AABB
   */
  std::vector<size_t> get_overlapping_cells(const search::AABB& aabb) const;

  /**
   * @brief Get candidate cells for point location
   */
  std::vector<index_t> get_candidate_cells(const std::array<real_t,3>& point) const;

  /**
   * @brief Get candidate vertices in grid cells near point
   */
  std::vector<index_t> get_nearby_vertices(const std::array<real_t,3>& point,
                                           real_t search_radius) const;

  /**
   * @brief Build grid for cells
   */
  void build_cell_grid(const MeshBase& mesh, Configuration cfg,
                      int resolution);

  /**
   * @brief Build grid for vertices
   */
  void build_vertex_grid(const MeshBase& mesh, Configuration cfg,
                        int resolution);

  /**
   * @brief Build boundary triangles for ray intersection
   */
  void build_boundary_triangles(const MeshBase& mesh, Configuration cfg);

  /**
   * @brief Check if point is in cell using cached data
   */
  bool point_in_cell_cached(const std::array<real_t,3>& point,
                           index_t cell_id,
                           const MeshBase& mesh) const;

  /**
   * @brief Walk through grid cells along ray
   */
  std::vector<size_t> walk_ray(const search::Ray& ray) const;
};

} // namespace svmp

#endif // SVMP_UNIFORM_GRID_ACCEL_H