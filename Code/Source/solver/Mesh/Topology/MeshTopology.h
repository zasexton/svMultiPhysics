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

#ifndef SVMP_MESH_TOPOLOGY_H
#define SVMP_MESH_TOPOLOGY_H

#include "../Core/MeshTypes.h"
#include <vector>
#include <unordered_set>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Mesh topology operations and adjacency queries
 *
 * This class provides topology-related operations including:
 * - Adjacency relationship construction (vertex-to-cell, cell-to-cell, etc.)
 * - Neighbor queries
 * - Boundary identification
 * - Connectivity analysis
 * - Topological invariants
 *
 * Notes on distributed meshes:
 * - All operations are rank-local and operate on the provided `MeshBase` data.
 * - No MPI communication or global reductions are performed here.
 * - In MPI runs, results depend on the local partition (and any ghost layer present).
 */
class MeshTopology {
public:
  // ---- Adjacency construction ----

  /**
   * @brief Build vertex-to-cell adjacency
   * @param mesh The mesh
   * @param[out] vertex2cell_offsets CSR offsets (size n_vertices+1)
   * @param[out] vertex2cell Concatenated cell indices
   */
  static void build_vertex2volume(const MeshBase& mesh,
                                  std::vector<offset_t>& vertex2cell_offsets,
                                  std::vector<index_t>& vertex2cell);

  /**
   * @brief Build vertex-to-face adjacency
   * @param mesh The mesh
   * @param[out] vertex2face_offsets CSR offsets (size n_vertices+1)
   * @param[out] vertex2face Concatenated face indices
   */
  static void build_vertex2codim1(const MeshBase& mesh,
                                  std::vector<offset_t>& vertex2entity_offsets,
                                  std::vector<index_t>& vertex2entity);

  /**
   * @brief Build cell-to-cell adjacency through shared faces
   * @param mesh The mesh
   * @param[out] cell2cell_offsets CSR offsets (size n_cells+1)
   * @param[out] cell2cell Concatenated neighbor cell indices
   */
  static void build_cell2cell(const MeshBase& mesh,
                              std::vector<offset_t>& cell2cell_offsets,
                              std::vector<index_t>& cell2cell);

  /**
   * @brief Build codim-1 to codim-1 adjacency through shared codim-2 entities
   * @param mesh The mesh
   * @param[out] entity2entity_offsets CSR offsets (size n_codim1+1)
   * @param[out] entity2entity Concatenated neighbor indices
   */
  static void build_codim1_to_codim1(const MeshBase& mesh,
                                     std::vector<offset_t>& entity2entity_offsets,
                                     std::vector<index_t>& entity2entity);

  // ---- Neighbor queries ----

  /**
   * @brief Get all cells sharing at least one vertex with given cell
   * @param mesh The mesh
   * @param cell Cell index
   * @param vertex2cell_offsets Prebuilt vertex-to-cell offsets (optional)
   * @param vertex2cell Prebuilt vertex-to-cell connectivity (optional)
   * @return Vector of neighbor cell indices
   */
  static std::vector<index_t> cell_neighbors(const MeshBase& mesh, index_t cell,
                                            const std::vector<offset_t>& cell2cell_offsets = {},
                                            const std::vector<index_t>& cell2cell = {});

  /**
   * @brief Get all cells containing given vertex
   * @param mesh The mesh
   * @param vertex Vertex index
   * @param vertex2cell_offsets Prebuilt vertex-to-cell offsets (optional)
   * @param vertex2cell Prebuilt vertex-to-cell connectivity (optional)
   * @return Vector of cell indices
   */
  static std::vector<index_t> vertex_cells(const MeshBase& mesh, index_t vertex,
                                          const std::vector<offset_t>& vertex2cell_offsets = {},
                                          const std::vector<index_t>& vertex2cell = {});

  /**
   * @brief Get all faces containing given vertex
   * @param mesh The mesh
   * @param vertex Vertex index
   * @param vertex2face_offsets Prebuilt vertex-to-face offsets (optional)
   * @param vertex2face Prebuilt vertex-to-face connectivity (optional)
   * @return Vector of face indices
   */
  static std::vector<index_t> vertex_codim1(const MeshBase& mesh, index_t vertex,
                                            const std::vector<offset_t>& vertex2entity_offsets = {},
                                            const std::vector<index_t>& vertex2entity = {});

  /**
   * @brief Get cells adjacent to given face
   * @param mesh The mesh
   * @param face Face index
   * @return Vector of 0, 1, or 2 cell indices
   */
  static std::vector<index_t> codim1_cells(const MeshBase& mesh, index_t entity);

  // ---- Boundary identification ----

  /**
   * @brief Get all boundary faces (faces with only one adjacent cell)
   * @param mesh The mesh
   * @return Vector of boundary face indices
   */
  static std::vector<index_t> boundary_codim1(const MeshBase& mesh);

  /**
   * @brief Get all boundary cells (cells with at least one boundary face)
   * @param mesh The mesh
   * @return Vector of boundary cell indices
   */
  static std::vector<index_t> boundary_cells(const MeshBase& mesh);

  /**
   * @brief Get all boundary vertices (vertices on boundary faces)
   * @param mesh The mesh
   * @return Vector of boundary vertex indices
   */
  static std::vector<index_t> boundary_vertices(const MeshBase& mesh);

  /**
   * @brief Check if a face is on the boundary
   * @param mesh The mesh
   * @param face Face index
   * @return True if face is on boundary
   */
  static bool is_boundary_codim1(const MeshBase& mesh, index_t entity);

  /**
   * @brief Check if a cell is on the boundary
   * @param mesh The mesh
   * @param cell Cell index
   * @return True if cell has at least one boundary face
   */
  static bool is_boundary_cell(const MeshBase& mesh, index_t cell);

  // ---- Connectivity analysis ----

  /**
   * @brief Find connected components of the mesh
   * @param mesh The mesh
   * @return Vector mapping each cell to its component ID
   */
  static std::vector<index_t> find_components(const MeshBase& mesh);

  /**
   * @brief Count number of connected components
   * @param mesh The mesh
   * @return Number of connected components
   */
  static index_t count_components(const MeshBase& mesh);

  /**
   * @brief Check if mesh is connected
   * @param mesh The mesh
   * @return True if mesh has only one connected component
   */
  static bool is_connected(const MeshBase& mesh);

  /**
   * @brief Find cells within distance of given cell
   * @param mesh The mesh
   * @param cell Starting cell
   * @param distance Maximum topological distance
   * @return Vector of cells within distance
   */
  static std::vector<index_t> cells_within_distance(const MeshBase& mesh,
                                                   index_t cell,
                                                   index_t distance);

  // ---- Topological features ----

  /**
   * @brief Count faces per cell type
   * @param mesh The mesh
   * @return Map from cell family to face count statistics
   */
  static std::unordered_map<CellFamily, std::pair<index_t, index_t>>
    face_count_by_cell_type(const MeshBase& mesh);

  /**
   * @brief Get valence (number of connected edges) for each vertex
   * @param mesh The mesh
   * @return Vector of valences indexed by vertex
   */
  static std::vector<index_t> vertex_valence(const MeshBase& mesh);

  /**
   * @brief Find vertices with irregular valence
   * @param mesh The mesh
   * @param expected_valence Expected valence for regular vertices
   * @return Vector of irregular vertex indices
   */
  static std::vector<index_t> irregular_vertices(const MeshBase& mesh,
                                                index_t expected_valence);

  /**
   * @brief Compute Euler characteristic
   * @param mesh The mesh
   * @return Euler characteristic (V - E + F - C for 3D)
   */
  static int euler_characteristic(const MeshBase& mesh);

  // ---- Edge operations ----

  /**
   * @brief Extract all unique edges from mesh topology
   * @param mesh The mesh
   * @return Vector of edge vertex pairs
   */
  static std::vector<std::array<index_t,2>> extract_edges(const MeshBase& mesh);

  /**
   * @brief Build edge-to-cell adjacency
   * @param mesh The mesh
   * @param edges Edge list
   * @param[out] edge2cell_offsets CSR offsets
   * @param[out] edge2cell Concatenated cell indices
   */
  static void build_edge2cell(const MeshBase& mesh,
                              const std::vector<std::array<index_t,2>>& edges,
                              std::vector<offset_t>& edge2cell_offsets,
                              std::vector<index_t>& edge2cell);

  /**
   * @brief Find boundary edges (edges on boundary faces)
   * @param mesh The mesh
   * @return Vector of boundary edge vertex pairs
   */
  static std::vector<std::array<index_t,2>> boundary_edges(const MeshBase& mesh);

  // ---- Manifold checks ----

  /**
   * @brief Check if mesh represents a manifold
   * @param mesh The mesh
   * @return True if mesh is manifold
   */
  static bool is_manifold(const MeshBase& mesh);

  /**
   * @brief Find non-manifold edges (shared by more than 2 faces)
   * @param mesh The mesh
   * @return Vector of non-manifold edge vertex pairs
   */
  static std::vector<std::array<index_t,2>> non_manifold_edges(const MeshBase& mesh);

  /**
   * @brief Find non-manifold vertices
   * @param mesh The mesh
   * @return Vector of non-manifold vertex indices
   */
  static std::vector<index_t> non_manifold_vertices(const MeshBase& mesh);
};

} // namespace svmp

#endif // SVMP_MESH_TOPOLOGY_H
