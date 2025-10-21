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

#ifndef SVMP_SEARCH_BUILDERS_H
#define SVMP_SEARCH_BUILDERS_H

#include "../Core/MeshTypes.h"
#include "SearchPrimitives.h"
#include <vector>
#include <array>
#include <memory>

namespace svmp {

// Forward declarations
class MeshBase;

namespace search {

/**
 * @brief Utilities for extracting and preparing mesh data for search structures
 */
class SearchBuilders {
public:
  // ---- Coordinate extraction ----

  /**
   * @brief Extract vertex coordinates for specified configuration
   * @param mesh The mesh
   * @param cfg Configuration (Reference or Current)
   * @return Vector of vertex coordinates
   */
  static std::vector<std::array<real_t,3>> extract_vertex_coords(
      const MeshBase& mesh,
      Configuration cfg);

  // Compatibility overloads used by some accelerators/tests
  static std::vector<std::array<real_t,3>> extract_vertex_coords(
      const MeshBase& mesh) { return extract_vertex_coords(mesh, Configuration::Reference); }

  static std::vector<std::array<real_t,3>> extract_deformed_coords(
      const MeshBase& mesh) { return extract_vertex_coords(mesh, Configuration::Current); }

  /**
   * @brief Extract coordinates for specific vertex
   * @param mesh The mesh
   * @param vertex_id Vertex index
   * @param cfg Configuration
   * @return Vertex coordinates
   */
  static std::array<real_t,3> get_vertex_coord(
      const MeshBase& mesh,
      index_t vertex_id,
      Configuration cfg);

  /**
   * @brief Extract coordinates for cell vertices
   * @param mesh The mesh
   * @param cell_id Cell index
   * @param cfg Configuration
   * @return Vector of vertex coordinates
   */
  static std::vector<std::array<real_t,3>> get_cell_vertex_coords(
      const MeshBase& mesh,
      index_t cell_id,
      Configuration cfg);

  /**
   * @brief Extract coordinates for face vertices
   * @param mesh The mesh
   * @param face_id Face index
   * @param cfg Configuration
   * @return Vector of vertex coordinates
   */
  static std::vector<std::array<real_t,3>> get_face_vertex_coords(
      const MeshBase& mesh,
      index_t face_id,
      Configuration cfg);

  // ---- AABB computation ----

  /**
   * @brief Compute AABB for entire mesh
   * @param mesh The mesh
   * @param cfg Configuration
   * @return Mesh bounding box
   */
  static AABB compute_mesh_aabb(const MeshBase& mesh, Configuration cfg);

  /**
   * @brief Compute AABB for a cell
   * @param mesh The mesh
   * @param cell_id Cell index
   * @param cfg Configuration
   * @return Cell bounding box
   */
  static AABB compute_cell_aabb(const MeshBase& mesh,
                                index_t cell_id,
                                Configuration cfg);

  /**
   * @brief Compute AABBs for all cells
   * @param mesh The mesh
   * @param cfg Configuration
   * @return Vector of cell AABBs
   */
  static std::vector<AABB> compute_all_cell_aabbs(const MeshBase& mesh,
                                                  Configuration cfg);

  // Overload: compute cell AABBs from provided vertex coordinates
  static std::vector<AABB> compute_cell_aabbs(const MeshBase& mesh,
                                              const std::vector<std::array<real_t,3>>& vertex_coords);

  /**
   * @brief Compute AABB for a face
   * @param mesh The mesh
   * @param face_id Face index
   * @param cfg Configuration
   * @return Face bounding box
   */
  static AABB compute_face_aabb(const MeshBase& mesh,
                                index_t face_id,
                                Configuration cfg);

  /**
   * @brief Compute AABBs for all boundary faces
   * @param mesh The mesh
   * @param cfg Configuration
   * @return Vector of boundary face AABBs
   */
  static std::vector<AABB> compute_boundary_face_aabbs(const MeshBase& mesh,
                                                       Configuration cfg);

  // ---- Boundary extraction ----

  /**
   * @brief Get indices of all boundary faces
   * @param mesh The mesh
   * @return Vector of boundary face indices
   */
  static std::vector<index_t> get_boundary_faces(const MeshBase& mesh);

  /**
   * @brief Check if a face is on the boundary
   * @param mesh The mesh
   * @param face_id Face index
   * @return True if face is on boundary
   */
  static bool is_boundary_face(const MeshBase& mesh, index_t face_id);

  /**
   * @brief Get boundary faces for a cell
   * @param mesh The mesh
   * @param cell_id Cell index
   * @return Vector of boundary face indices for this cell
   */
  static std::vector<index_t> get_cell_boundary_faces(const MeshBase& mesh,
                                                      index_t cell_id);

  // ---- Face triangulation ----

  /**
   * @brief Triangulate a face (for non-triangular faces)
   * @param mesh The mesh
   * @param face_id Face index
   * @param cfg Configuration
   * @return Vector of triangles (each triangle is 3 vertex coords)
   */
  static std::vector<std::array<std::array<real_t,3>,3>> triangulate_face(
      const MeshBase& mesh,
      index_t face_id,
      Configuration cfg);

  /**
   * @brief Triangulate all boundary faces
   * @param mesh The mesh
   * @param cfg Configuration
   * @return Vector of triangles with face indices
   */
  struct TriangleWithFace {
    std::array<std::array<real_t,3>,3> vertices;
    index_t face_id;
  };

  static std::vector<TriangleWithFace> triangulate_boundary(
      const MeshBase& mesh,
      Configuration cfg);

  // Extract boundary triangles using precomputed vertex coordinates
  static std::vector<TriangleWithFace> extract_boundary_triangles(
      const MeshBase& mesh,
      const std::vector<std::array<real_t,3>>& vertex_coords);

  // Point-in-cell convenience (returns also parametric coords)
  static bool point_in_cell(const MeshBase& mesh,
                            const std::vector<std::array<real_t,3>>& vertex_coords,
                            index_t cell_id,
                            const std::array<real_t,3>& p,
                            std::array<real_t,3>& xi);

  // ---- Cell center computation ----

  /**
   * @brief Compute center of a cell
   * @param mesh The mesh
   * @param cell_id Cell index
   * @param cfg Configuration
   * @return Cell center coordinates
   */
  static std::array<real_t,3> compute_cell_center(const MeshBase& mesh,
                                                  index_t cell_id,
                                                  Configuration cfg);

  /**
   * @brief Compute centers for all cells
   * @param mesh The mesh
   * @param cfg Configuration
   * @return Vector of cell centers
   */
  static std::vector<std::array<real_t,3>> compute_all_cell_centers(
      const MeshBase& mesh,
      Configuration cfg);

  // ---- Neighbor information ----

  /**
   * @brief Get neighboring cells for a cell
   * @param mesh The mesh
   * @param cell_id Cell index
   * @return Vector of neighbor cell indices
   */
  static std::vector<index_t> get_cell_neighbors(const MeshBase& mesh,
                                                 index_t cell_id);

  /**
   * @brief Get cells sharing a vertex
   * @param mesh The mesh
   * @param vertex_id Vertex index
   * @return Vector of cell indices
   */
  static std::vector<index_t> get_vertex_cells(const MeshBase& mesh,
                                               index_t vertex_id);

  /**
   * @brief Get cells sharing an edge
   * @param mesh The mesh
   * @param v0 First vertex of edge
   * @param v1 Second vertex of edge
   * @return Vector of cell indices
   */
  static std::vector<index_t> get_edge_cells(const MeshBase& mesh,
                                             index_t v0,
                                             index_t v1);

  // ---- Parametric coordinate helpers ----

  /**
   * @brief Compute parametric coordinates for point in tetrahedron
   * @param p Physical point
   * @param tet_vertices Tetrahedron vertices
   * @return Parametric coordinates (xi, eta, zeta)
   */
  static std::array<real_t,3> tetra_parametric_coords(
      const std::array<real_t,3>& p,
      const std::vector<std::array<real_t,3>>& tet_vertices);

  /**
   * @brief Compute parametric coordinates for point in hexahedron
   * @param p Physical point
   * @param hex_vertices Hexahedron vertices (8 vertices)
   * @param max_iter Maximum Newton iterations
   * @param tol Convergence tolerance
   * @return Parametric coordinates (xi, eta, zeta)
   */
  static std::array<real_t,3> hex_parametric_coords(
      const std::array<real_t,3>& p,
      const std::vector<std::array<real_t,3>>& hex_vertices,
      int max_iter = 20,
      real_t tol = 1e-10);

  /**
   * @brief Generic parametric coordinate computation
   * @param mesh The mesh
   * @param cell_id Cell index
   * @param p Physical point
   * @param cfg Configuration
   * @return Parametric coordinates
   */
  static std::array<real_t,3> compute_parametric_coords(
      const MeshBase& mesh,
      index_t cell_id,
      const std::array<real_t,3>& p,
      Configuration cfg);

  // ---- Mesh statistics ----

  /**
   * @brief Compute mesh characteristic length
   * @param mesh The mesh
   * @param cfg Configuration
   * @return Characteristic length (e.g., average edge length)
   */
  static real_t compute_mesh_characteristic_length(const MeshBase& mesh,
                                                   Configuration cfg);

  /**
   * @brief Estimate optimal grid resolution for uniform grid
   * @param mesh The mesh
   * @param cfg Configuration
   * @param target_cells_per_bucket Target number of cells per grid bucket
   * @return Suggested grid resolution
   */
  static int estimate_grid_resolution(const MeshBase& mesh,
                                      Configuration cfg,
                                      int target_cells_per_bucket = 10);

  /**
   * @brief Check if mesh is small enough for linear search
   * @param mesh The mesh
   * @param threshold Cell count threshold
   * @return True if linear search is recommended
   */
  static bool use_linear_search(const MeshBase& mesh,
                                size_t threshold = 100);
};

} // namespace search
} // namespace svmp

#endif // SVMP_SEARCH_BUILDERS_H
