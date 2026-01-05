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

#ifndef SVMP_MESH_SEARCH_H
#define SVMP_MESH_SEARCH_H

#include "../Core/MeshTypes.h"
#include "../Topology/CellShape.h"
#include <memory>
#include <vector>
#include <array>

namespace svmp {

// Forward declaration
class MeshBase;
class DistributedMesh;

/**
 * @brief Spatial search and point location for meshes
 *
 * This class provides:
 * - Point location (find cell containing point)
 * - Nearest neighbor searches
 * - Ray-mesh intersection
 * - Distance queries
 * - Spatial acceleration structures
 */
class MeshSearch {
public:
  // ---- Search structures ----

  /**
   * @brief Types of search acceleration structures
   */
  enum class AccelType {
    None,           // No acceleration (linear search)
    UniformGrid,    // Uniform spatial grid
    Octree,         // Octree/Quadtree
    KDTree,         // K-d tree
    BVH,           // Bounding volume hierarchy
    RTree          // R-tree
  };

  /**
   * @brief Search structure configuration
   */
  struct SearchConfig {
    AccelType type = AccelType::UniformGrid;
    int max_depth = 10;              // Maximum tree depth
    int min_entities_per_leaf = 1;   // Minimum entities in leaf
    int grid_resolution = 32;        // Grid cells per dimension
    real_t tolerance = 1e-10;        // Geometric tolerance
    bool use_cache = true;           // Cache recent searches
    // Primary query type (compat with some accelerators/tests)
    enum class QueryType {
      PointLocation,
      NearestNeighbor,
      RayIntersection
    };
    QueryType primary_use = QueryType::PointLocation;
  };

  // Compatibility alias so tests can use MeshSearch::QueryType
  using QueryType = SearchConfig::QueryType;

  // ---- Point location ----

  /**
   * @brief Locate point in mesh (find containing cell)
   * @param mesh The mesh
   * @param point Query point
   * @param cfg Configuration (Reference or Current)
   * @param hint_cell Optional hint for starting cell
   * @return Point location result
   */
  static PointLocateResult locate_point(const MeshBase& mesh,
                                       const std::array<real_t,3>& point,
                                       Configuration cfg = Configuration::Reference,
                                       index_t hint_cell = -1);

  /**
   * @brief Locate point in a distributed mesh (collective in MPI builds).
   *
   * Semantics:
   * - In serial builds (or single-rank runs), this is equivalent to locate_point(mesh.local_mesh(), ...).
   * - In MPI builds, all ranks in the mesh communicator must call this with the same input point.
   *   The owning rank returns a valid local cell_id; other ranks return found==true with cell_id==INVALID_INDEX.
   */
  static PointLocateResult locate_point_global(const DistributedMesh& mesh,
                                              const std::array<real_t,3>& point,
                                              Configuration cfg = Configuration::Reference);

  /**
   * @brief Locate multiple points efficiently
   * @param mesh The mesh
   * @param points Query points
   * @param cfg Configuration
   * @return Vector of location results
   */
  static std::vector<PointLocateResult> locate_points(const MeshBase& mesh,
                                                     const std::vector<std::array<real_t,3>>& points,
                                                     Configuration cfg = Configuration::Reference);

  /**
   * @brief Locate multiple points in a distributed mesh (collective in MPI builds).
   *
   * All ranks must pass the same `points` vector in MPI builds.
   */
  static std::vector<PointLocateResult> locate_points_global(
      const DistributedMesh& mesh,
      const std::vector<std::array<real_t,3>>& points,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Check if point is inside mesh domain
   * @param mesh The mesh
   * @param point Query point
   * @param cfg Configuration
   * @return True if point is inside mesh
   */
  static bool contains_point(const MeshBase& mesh,
                            const std::array<real_t,3>& point,
                            Configuration cfg = Configuration::Reference);

  /**
   * @brief Check if point is inside distributed mesh domain (collective in MPI builds).
   */
  static bool contains_point_global(const DistributedMesh& mesh,
                                    const std::array<real_t,3>& point,
                                    Configuration cfg = Configuration::Reference);

  // ---- Nearest neighbor search ----

  /**
   * @brief Find nearest vertex to point
   * @param mesh The mesh
   * @param point Query point
   * @param cfg Configuration
   * @return Vertex index and distance
   */
  static std::pair<index_t, real_t> nearest_vertex(const MeshBase& mesh,
                                                  const std::array<real_t,3>& point,
                                                  Configuration cfg = Configuration::Reference);

  /**
   * @brief Find k nearest vertices to point
   * @param mesh The mesh
   * @param point Query point
   * @param k Number of neighbors
   * @param cfg Configuration
   * @return Vector of (vertex_index, distance) pairs
   */
  static std::vector<std::pair<index_t, real_t>> k_nearest_vertices(
      const MeshBase& mesh,
      const std::array<real_t,3>& point,
      size_t k,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Find all vertices within radius
   * @param mesh The mesh
   * @param point Query point
   * @param radius Search radius
   * @param cfg Configuration
   * @return Vector of vertex indices
   */
  static std::vector<index_t> vertices_in_radius(const MeshBase& mesh,
                                                const std::array<real_t,3>& point,
                                                real_t radius,
                                                Configuration cfg = Configuration::Reference);

  /**
   * @brief Find nearest cell to point
   * @param mesh The mesh
   * @param point Query point
   * @param cfg Configuration
   * @return Cell index and distance
   */
  static std::pair<index_t, real_t> nearest_cell(const MeshBase& mesh,
                                                const std::array<real_t,3>& point,
                                                Configuration cfg = Configuration::Reference);

  // ---- Ray intersection ----

  /**
   * @brief Find ray-mesh intersection
   * @param mesh The mesh
   * @param origin Ray origin
   * @param direction Ray direction (normalized)
   * @param cfg Configuration
   * @param max_distance Maximum ray distance
   * @return Ray intersection result
   */
  static RayIntersectResult intersect_ray(const MeshBase& mesh,
                                         const std::array<real_t,3>& origin,
                                         const std::array<real_t,3>& direction,
                                         Configuration cfg = Configuration::Reference,
                                         real_t max_distance = 1e300);

  /**
   * @brief Find the first ray-mesh intersection on a distributed mesh (collective in MPI builds).
   *
   * In MPI builds, all ranks must call this with identical inputs. The intersection is performed
   * against the *true global boundary* (not partition interfaces), even when no ghost layer is present.
   *
   * On the winning rank, `face_id` is a local face index. On all other ranks, `face_id==INVALID_INDEX`.
   */
  static RayIntersectResult intersect_ray_global(const DistributedMesh& mesh,
                                                const std::array<real_t,3>& origin,
                                                const std::array<real_t,3>& direction,
                                                Configuration cfg = Configuration::Reference,
                                                real_t max_distance = 1e300);

  /**
   * @brief Find all ray-mesh intersections
   * @param mesh The mesh
   * @param origin Ray origin
   * @param direction Ray direction
   * @param cfg Configuration
   * @param max_distance Maximum ray distance
   * @return Vector of intersection results
   */
  static std::vector<RayIntersectResult> intersect_ray_all(
      const MeshBase& mesh,
      const std::array<real_t,3>& origin,
      const std::array<real_t,3>& direction,
      Configuration cfg = Configuration::Reference,
      real_t max_distance = 1e300);

  /**
   * @brief Find all ray-mesh intersections on a distributed mesh (collective in MPI builds).
   *
   * All ranks must pass identical inputs in MPI builds. The returned list is only populated on
   * rank 0; other ranks return an empty vector.
   */
  static std::vector<RayIntersectResult> intersect_ray_all_global(
      const DistributedMesh& mesh,
      const std::array<real_t,3>& origin,
      const std::array<real_t,3>& direction,
      Configuration cfg = Configuration::Reference,
      real_t max_distance = 1e300);

  // ---- Distance queries ----

  /**
   * @brief Compute signed distance to mesh boundary
   * @param mesh The mesh
   * @param point Query point
   * @param cfg Configuration
   * @return Signed distance (negative inside, positive outside)
   */
  static real_t signed_distance(const MeshBase& mesh,
                               const std::array<real_t,3>& point,
                               Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute signed distance to the true global boundary on a distributed mesh (collective in MPI builds).
   */
  static real_t signed_distance_global(const DistributedMesh& mesh,
                                       const std::array<real_t,3>& point,
                                       Configuration cfg = Configuration::Reference);

  /**
   * @brief Find closest point on mesh boundary
   * @param mesh The mesh
   * @param point Query point
   * @param cfg Configuration
   * @return Closest point and face index
   */
  static std::pair<std::array<real_t,3>, index_t> closest_boundary_point(
      const MeshBase& mesh,
      const std::array<real_t,3>& point,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Find closest point on the true global boundary of a distributed mesh (collective in MPI builds).
   *
   * On the winning rank, `index_t` is a local boundary-entity id (typically a face index for 3D meshes).
   * On other ranks, `index_t==INVALID_INDEX`.
   */
  static std::pair<std::array<real_t,3>, index_t> closest_boundary_point_global(
      const DistributedMesh& mesh,
      const std::array<real_t,3>& point,
      Configuration cfg = Configuration::Reference);

  // ---- Search structure management ----

  /**
   * @brief Build search acceleration structure
   * @param mesh The mesh
   * @param config Search configuration
   * @param cfg Mesh configuration
   */
  static void build_search_structure(const MeshBase& mesh,
                                    const SearchConfig& config,
                                    Configuration cfg = Configuration::Reference);

  /**
   * @brief Build search acceleration structure with default config
   * @param mesh The mesh
   * @param cfg Mesh configuration
   */
  static void build_search_structure(const MeshBase& mesh,
                                    Configuration cfg = Configuration::Reference) {
    SearchConfig config;
    build_search_structure(mesh, config, cfg);
  }

  /**
   * @brief Clear search structure
   * @param mesh The mesh
   */
  static void clear_search_structure(const MeshBase& mesh);

  /**
   * @brief Check if search structure is built
   * @param mesh The mesh
   * @return True if structure exists
   */
  static bool has_search_structure(const MeshBase& mesh);

  // ---- Spatial queries ----

  /**
   * @brief Find cells intersecting axis-aligned box
   * @param mesh The mesh
   * @param box_min Minimum corner
   * @param box_max Maximum corner
   * @param cfg Configuration
   * @return Vector of cell indices
   */
  static std::vector<index_t> cells_in_box(const MeshBase& mesh,
                                          const std::array<real_t,3>& box_min,
                                          const std::array<real_t,3>& box_max,
                                          Configuration cfg = Configuration::Reference);

  /**
   * @brief Find cells intersecting sphere
   * @param mesh The mesh
   * @param center Sphere center
   * @param radius Sphere radius
   * @param cfg Configuration
   * @return Vector of cell indices
   */
  static std::vector<index_t> cells_in_sphere(const MeshBase& mesh,
                                             const std::array<real_t,3>& center,
                                             real_t radius,
                                             Configuration cfg = Configuration::Reference);

  // ---- Parametric coordinates ----

  /**
   * @brief Compute parametric coordinates of point in cell
   * @param mesh The mesh
   * @param cell Cell index
   * @param point Physical point
   * @param cfg Configuration
   * @return Parametric coordinates (xi, eta, zeta)
   */
  static std::array<real_t,3> compute_parametric_coords(const MeshBase& mesh,
                                                       index_t cell,
                                                       const std::array<real_t,3>& point,
                                                       Configuration cfg = Configuration::Reference);

  /**
   * @brief Check if parametric coordinates are inside reference element
   * @param shape Cell shape
   * @param xi Parametric coordinates
   * @return True if inside
   */
  static bool is_inside_reference_element(const CellShape& shape,
                                         const std::array<real_t,3>& xi);

  // ---- Walking algorithms ----

  /**
   * @brief Walk from cell to cell containing point
   * @param mesh The mesh
   * @param start_cell Starting cell
   * @param target_point Target point
   * @param cfg Configuration
   * @return Path of cells traversed
   */
  static std::vector<index_t> walk_to_point(const MeshBase& mesh,
                                           index_t start_cell,
                                           const std::array<real_t,3>& target_point,
                                           Configuration cfg = Configuration::Reference);

private:
  // Internal search acceleration structure
  struct SearchAccel;

  // Helper methods
  static bool point_in_cell(const MeshBase& mesh,
                           const std::array<real_t,3>& point,
                           index_t cell,
                           Configuration cfg);

  static real_t distance_to_cell(const MeshBase& mesh,
                                const std::array<real_t,3>& point,
                                index_t cell,
                                Configuration cfg);
};

} // namespace svmp

#endif // SVMP_MESH_SEARCH_H
