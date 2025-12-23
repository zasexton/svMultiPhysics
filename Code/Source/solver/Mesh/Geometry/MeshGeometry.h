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

#ifndef SVMP_MESH_GEOMETRY_H
#define SVMP_MESH_GEOMETRY_H

#include "../Core/MeshTypes.h"
#include "../Topology/CellShape.h"
#include <array>
#include <vector>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Mesh geometry computations
 *
 * This class provides geometric computations for mesh entities including:
 * - Centers (cell, face, edge)
 * - Normals (face, edge)
 * - Measures (length, area, volume)
 * - Bounding boxes
 * - Geometric transformations
 */
class MeshGeometry {
public:
  // ---- Centers and bounding boxes ----

  /**
   * @brief Compute a vertex-average center of a cell
   * @param mesh The mesh
   * @param cell Cell index
   * @param cfg Reference or current configuration
   * @return Vertex-average center coordinates
   *
   * Notes:
   * - For simplices (line/tri/tet) this matches the geometric centroid.
   * - For general polyhedra and pyramids this differs from the volume centroid.
   */
  static std::array<real_t,3> cell_center(const MeshBase& mesh, index_t cell,
                                         Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute geometric centroid of a cell in its topological dimension
   * @param mesh The mesh
   * @param cell Cell index
   * @param cfg Reference or current configuration
   * @return Cell centroid coordinates
   *
   * For 3D elements, this is the volume centroid; for 2D elements, the area centroid.
   * For polyhedra, this requires explicit face connectivity and assumes a convex, watertight cell.
   */
  static std::array<real_t,3> cell_centroid(const MeshBase& mesh, index_t cell,
                                           Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute center of a face
   * @param mesh The mesh
   * @param face Face index
   * @param cfg Reference or current configuration
   * @return Face center coordinates
   */
  static std::array<real_t,3> face_center(const MeshBase& mesh, index_t face,
                                         Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute center of an edge
   * @param mesh The mesh
   * @param edge Edge index
   * @param cfg Reference or current configuration
   * @return Edge midpoint coordinates
   */
  static std::array<real_t,3> edge_center(const MeshBase& mesh, index_t edge,
                                         Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute bounding box of the entire mesh
   * @param mesh The mesh
   * @param cfg Reference or current configuration
   * @return Bounding box
   */
  static BoundingBox bounding_box(const MeshBase& mesh,
                                 Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute bounding box of a cell
   * @param mesh The mesh
   * @param cell Cell index
   * @param cfg Reference or current configuration
   * @return Cell bounding box
   */
  static BoundingBox cell_bounding_box(const MeshBase& mesh, index_t cell,
                                      Configuration cfg = Configuration::Reference);

  // ---- Normals ----

  /**
   * @brief Compute face normal vector
   * @param mesh The mesh
   * @param face Face index
   * @param cfg Reference or current configuration
   * @return Unit normal vector (or zero if degenerate)
   */
  static std::array<real_t,3> face_normal(const MeshBase& mesh, index_t face,
                                         Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute face normal vector (unnormalized)
   * @param mesh The mesh
   * @param face Face index
   * @param cfg Reference or current configuration
   * @return Normal vector with magnitude = 2*area
   */
  static std::array<real_t,3> face_normal_unnormalized(const MeshBase& mesh, index_t face,
                                                      Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute edge normal
   * @param mesh The mesh
   * @param edge Edge index
   * @param cfg Reference or current configuration
   * @return Unit normal vector perpendicular to edge
   *
   * For 2D meshes: Returns normal in xy-plane (z=0)
   * For 3D edges: Returns a perpendicular vector
   */
  static std::array<real_t,3> edge_normal(const MeshBase& mesh, index_t edge,
                                         Configuration cfg = Configuration::Reference);

  // ---- Measures (length, area, volume) ----

  /**
   * @brief Compute cell measure (length/area/volume depending on dimension)
   * @param mesh The mesh
   * @param cell Cell index
   * @param cfg Reference or current configuration
   * @return Cell measure
   */
  static real_t cell_measure(const MeshBase& mesh, index_t cell,
                            Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute face area (or length in 2D)
   * @param mesh The mesh
   * @param face Face index
   * @param cfg Reference or current configuration
   * @return Face area/length
   */
  static real_t face_area(const MeshBase& mesh, index_t face,
                        Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute edge length
   * @param mesh The mesh
   * @param edge Edge index
   * @param cfg Reference or current configuration
   * @return Edge length
   */
  static real_t edge_length(const MeshBase& mesh, index_t edge,
                          Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute total mesh volume (or area in 2D)
   * @param mesh The mesh
   * @param cfg Reference or current configuration
   * @return Total volume/area
   */
  static real_t total_volume(const MeshBase& mesh,
                           Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute total surface area of boundary faces
   * @param mesh The mesh
   * @param cfg Reference or current configuration
   * @return Total boundary area
   */
  static real_t boundary_area(const MeshBase& mesh,
                            Configuration cfg = Configuration::Reference);

  // ---- Boundary-specific geometry ----

  /**
   * @brief Compute normal from oriented vertex list (right-hand rule)
   * @param mesh The mesh
   * @param oriented_vertices Vertices in right-hand rule order
   * @param cfg Reference or current configuration
   * @return Normal vector (unnormalized) pointing outward
   */
  static std::array<real_t,3> compute_normal_from_vertices(
      const MeshBase& mesh,
      const std::vector<index_t>& oriented_vertices,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute edge normal from oriented vertex list
   * @param mesh The mesh
   * @param oriented_vertices Edge vertices in order
   * @param cfg Reference or current configuration
   * @return Normal vector (unnormalized, perpendicular to edge)
   *
   * For 2D meshes: Returns normal in xy-plane (z=0)
   * For 3D edges: Returns a perpendicular vector (additional context may be needed for unique direction)
   */
  static std::array<real_t,3> compute_edge_normal_from_vertices(
      const MeshBase& mesh,
      const std::vector<index_t>& oriented_vertices,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute area from oriented vertex list
   * @param mesh The mesh
   * @param oriented_vertices Vertices defining the boundary
   * @param cfg Reference or current configuration
   * @return Area (or length for 1D boundaries)
   */
  static real_t compute_area_from_vertices(
      const MeshBase& mesh,
      const std::vector<index_t>& oriented_vertices,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute centroid from vertex indices
   * @param mesh The mesh
   * @param vertices Vertex indices
   * @param cfg Reference or current configuration
   * @return Centroid coordinates
   */
  static std::array<real_t,3> compute_centroid_from_vertices(
      const MeshBase& mesh,
      const std::vector<index_t>& vertices,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Compute bounding box from vertex indices
   * @param mesh The mesh
   * @param vertices Vertex indices
   * @param cfg Reference or current configuration
   * @return Bounding box
   */
  static BoundingBox compute_bounding_box_from_vertices(
      const MeshBase& mesh,
      const std::vector<index_t>& vertices,
      Configuration cfg = Configuration::Reference);

  // ---- Specialized shape measures ----

  /**
   * @brief Compute tetrahedron volume using scalar triple product
   * @param p0, p1, p2, p3 Tetrahedron vertices
   * @return Signed volume (positive if vertices are ordered correctly)
   */
  static real_t tet_volume(const std::array<real_t,3>& p0,
                         const std::array<real_t,3>& p1,
                         const std::array<real_t,3>& p2,
                         const std::array<real_t,3>& p3);

  /**
   * @brief Compute triangle area using cross product
   * @param p0, p1, p2 Triangle vertices
   * @return Triangle area
   */
  static real_t triangle_area(const std::array<real_t,3>& p0,
                             const std::array<real_t,3>& p1,
                             const std::array<real_t,3>& p2);

  /**
   * @brief Compute hexahedron volume via tetrahedral decomposition
   * @param vertices 8 hex vertices in standard ordering
   * @return Hex volume
   */
  static real_t hex_volume(const std::vector<std::array<real_t,3>>& vertices);

  /**
   * @brief Compute wedge (prism) volume
   * @param vertices 6 wedge vertices in standard ordering
   * @return Wedge volume
   */
  static real_t wedge_volume(const std::vector<std::array<real_t,3>>& vertices);

  /**
   * @brief Compute pyramid volume
   * @param vertices 5 pyramid vertices (base quad + apex)
   * @return Pyramid volume
   */
  static real_t pyramid_volume(const std::vector<std::array<real_t,3>>& vertices);

  /**
   * @brief Compute quadrilateral area (planar or warped)
   * @param vertices 4 quad vertices
   * @return Quad area
   */
  static real_t quad_area(const std::vector<std::array<real_t,3>>& vertices);

  /**
   * @brief Compute general polygon area
   * @param vertices Polygon vertices (coplanar assumed)
   * @return Polygon area
   */
  static real_t polygon_area(const std::vector<std::array<real_t,3>>& vertices);

  // ---- Distance and angle computations ----

  /**
   * @brief Compute distance between two points
   * @param p1, p2 Points
   * @return Euclidean distance
   */
  static real_t distance(const std::array<real_t,3>& p1,
                        const std::array<real_t,3>& p2);

  /**
   * @brief Compute angle between three points (at middle point)
   * @param p1, p2, p3 Points forming angle at p2
   * @return Angle in radians
   */
  static real_t angle(const std::array<real_t,3>& p1,
                     const std::array<real_t,3>& p2,
                     const std::array<real_t,3>& p3);

  /**
   * @brief Compute dihedral angle between two faces sharing an edge
   * @param n1, n2 Face normals
   * @return Dihedral angle in radians
   */
  static real_t dihedral_angle(const std::array<real_t,3>& n1,
                              const std::array<real_t,3>& n2);

  // ---- Vector operations ----

  /**
   * @brief Cross product of two 3D vectors
   * @param a, b Input vectors
   * @return a × b
   */
  static std::array<real_t,3> cross(const std::array<real_t,3>& a,
                                   const std::array<real_t,3>& b);

  /**
   * @brief Dot product of two 3D vectors
   * @param a, b Input vectors
   * @return a · b
   */
  static real_t dot(const std::array<real_t,3>& a,
                   const std::array<real_t,3>& b);

  /**
   * @brief Normalize a 3D vector
   * @param v Input vector
   * @return Unit vector (or zero if input is zero)
   */
  static std::array<real_t,3> normalize(const std::array<real_t,3>& v);

  /**
   * @brief Compute vector magnitude
   * @param v Input vector
   * @return |v|
   */
  static real_t magnitude(const std::array<real_t,3>& v);

private:
  // Helper to get coordinates from mesh
  static std::vector<std::array<real_t,3>> get_cell_vertices(const MeshBase& mesh,
                                                            index_t cell,
                                                            Configuration cfg);

  static std::vector<std::array<real_t,3>> get_face_vertices(const MeshBase& mesh,
                                                            index_t face,
                                                            Configuration cfg);
};

} // namespace svmp

#endif // SVMP_MESH_GEOMETRY_H
