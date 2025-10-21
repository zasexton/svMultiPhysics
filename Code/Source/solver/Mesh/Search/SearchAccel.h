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

#ifndef SVMP_SEARCH_ACCEL_H
#define SVMP_SEARCH_ACCEL_H

#include "../Core/MeshTypes.h"
#include "MeshSearch.h"
#include <memory>
#include <vector>
#include <array>
#include <chrono>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Statistics for search acceleration structure
 */
struct SearchStats {
  size_t build_time_ms = 0;         // Build time in milliseconds
  size_t memory_bytes = 0;          // Memory usage in bytes
  size_t query_count = 0;           // Total queries performed
  size_t hit_count = 0;             // Successful queries
  size_t n_entities = 0;            // Number of entities in structure
  size_t tree_depth = 0;            // Tree depth (for tree structures)
  // Extended counters used by some accelerators
  size_t n_nodes = 0;               // Node count (trees/BVH)
  size_t n_node_visits = 0;         // Node visits across queries
  double total_query_time_ms = 0.0; // Aggregate query time
  // Compatibility alias (used by some modules/tests)
  size_t n_queries = 0;             // Total queries performed

  real_t hit_ratio() const {
    return query_count > 0 ? static_cast<real_t>(hit_count) / query_count : 0.0;
  }
};

/**
 * @brief Abstract interface for spatial acceleration structures
 */
class IAccel {
public:
  // Compatibility alias for configuration enum used in tests
  using Configuration = svmp::Configuration;
  virtual ~IAccel() = default;

  // ---- Building ----

  /**
   * @brief Build acceleration structure
   * @param mesh The mesh to build from
   * @param cfg Configuration (Reference or Current)
   * @param config Search configuration
   */
  virtual void build(const MeshBase& mesh,
                    Configuration cfg,
                    const MeshSearch::SearchConfig& config) = 0;

  /**
   * @brief Clear the acceleration structure
   */
  virtual void clear() = 0;

  /**
   * @brief Check if structure is built
   */
  virtual bool is_built() const = 0;

  /**
   * @brief Get configuration the structure was built for
   */
  virtual Configuration built_config() const = 0;

  // ---- Point location ----

  /**
   * @brief Locate point in mesh
   * @param mesh The mesh
   * @param point Query point
   * @param hint_cell Optional hint cell
   * @return Point location result
   */
  virtual PointLocateResult locate_point(const MeshBase& mesh,
                                        const std::array<real_t,3>& point,
                                        index_t hint_cell = -1) const = 0;

  /**
   * @brief Locate multiple points (batched for efficiency)
   * @param mesh The mesh
   * @param points Query points
   * @return Vector of location results
   */
  virtual std::vector<PointLocateResult> locate_points(
      const MeshBase& mesh,
      const std::vector<std::array<real_t,3>>& points) const = 0;

  // ---- Nearest neighbor ----

  /**
   * @brief Find nearest vertex
   * @param mesh The mesh
   * @param point Query point
   * @return Vertex index and distance
   */
  virtual std::pair<index_t, real_t> nearest_vertex(
      const MeshBase& mesh,
      const std::array<real_t,3>& point) const = 0;

  /**
   * @brief Find k nearest vertices
   * @param mesh The mesh
   * @param point Query point
   * @param k Number of neighbors
   * @return Vector of (vertex_index, distance) pairs
   */
  virtual std::vector<std::pair<index_t, real_t>> k_nearest_vertices(
      const MeshBase& mesh,
      const std::array<real_t,3>& point,
      size_t k) const = 0;

  /**
   * @brief Find vertices within radius
   * @param mesh The mesh
   * @param point Query point
   * @param radius Search radius
   * @return Vector of vertex indices
   */
  virtual std::vector<index_t> vertices_in_radius(
      const MeshBase& mesh,
      const std::array<real_t,3>& point,
      real_t radius) const = 0;

  // ---- Ray intersection ----

  /**
   * @brief Find first ray-mesh intersection
   * @param mesh The mesh
   * @param origin Ray origin
   * @param direction Ray direction (normalized)
   * @param max_distance Maximum ray distance
   * @return Ray intersection result
   */
  virtual RayIntersectResult intersect_ray(
      const MeshBase& mesh,
      const std::array<real_t,3>& origin,
      const std::array<real_t,3>& direction,
      real_t max_distance = 1e300) const = 0;

  /**
   * @brief Find all ray-mesh intersections
   * @param mesh The mesh
   * @param origin Ray origin
   * @param direction Ray direction
   * @param max_distance Maximum ray distance
   * @return Vector of intersection results
   */
  virtual std::vector<RayIntersectResult> intersect_ray_all(
      const MeshBase& mesh,
      const std::array<real_t,3>& origin,
      const std::array<real_t,3>& direction,
      real_t max_distance = 1e300) const = 0;

  // ---- Region queries ----

  /**
   * @brief Find cells intersecting axis-aligned box
   * @param mesh The mesh
   * @param box_min Minimum corner
   * @param box_max Maximum corner
   * @return Vector of cell indices
   */
  virtual std::vector<index_t> cells_in_box(
      const MeshBase& mesh,
      const std::array<real_t,3>& box_min,
      const std::array<real_t,3>& box_max) const = 0;

  /**
   * @brief Find cells intersecting sphere
   * @param mesh The mesh
   * @param center Sphere center
   * @param radius Sphere radius
   * @return Vector of cell indices
   */
  virtual std::vector<index_t> cells_in_sphere(
      const MeshBase& mesh,
      const std::array<real_t,3>& center,
      real_t radius) const = 0;

  // ---- Statistics ----

  /**
   * @brief Get structure statistics
   */
  virtual SearchStats get_stats() const = 0;
};

/**
 * @brief Search acceleration structure holder (PIMPL for MeshBase)
 */
struct SearchAccel {
  std::unique_ptr<IAccel> accel;      // The acceleration structure
  Configuration built_cfg;             // Configuration it was built for
  std::array<real_t,6> bounds;        // Bounding box [xmin,ymin,zmin,xmax,ymax,zmax]
  SearchStats stats;                   // Performance statistics

  SearchAccel() : built_cfg(Configuration::Reference) {
    bounds.fill(0.0);
  }

  bool is_valid() const {
    return accel && accel->is_built();
  }

  void invalidate() {
    if (accel) {
      accel->clear();
    }
    accel.reset();
    stats = SearchStats();
  }
};

/**
 * @brief Factory for creating acceleration structures
 */
class AccelFactory {
public:
  /**
   * @brief Create an acceleration structure
   * @param type Type of structure to create
   * @return Unique pointer to acceleration structure
   */
  static std::unique_ptr<IAccel> create(MeshSearch::AccelType type);
};

} // namespace svmp

#endif // SVMP_SEARCH_ACCEL_H
