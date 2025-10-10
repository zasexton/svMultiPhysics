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

#ifndef SVMP_MESH_BUILDERS_H
#define SVMP_MESH_BUILDERS_H

#include "../Core/MeshTypes.h"
#include <memory>
#include <functional>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Mesh construction utilities
 *
 * This class provides various mesh builders and generators:
 * - Structured mesh generation (Cartesian, cylindrical, spherical)
 * - Mesh extrusion
 * - Mesh merging and combination
 * - Primitive shape generation
 * - Mesh transformation
 */
class MeshBuilders {
public:
  // ---- Structured mesh generation ----

  /**
   * @brief Build Cartesian grid mesh
   * @param nx, ny, nz Number of cells in each direction
   * @param domain Bounding box for the mesh
   * @return Generated mesh
   */
  static std::unique_ptr<MeshBase> build_cartesian(
      int nx, int ny, int nz,
      const BoundingBox& domain);

  /**
   * @brief Build 2D Cartesian grid
   * @param nx, ny Number of cells in x, y directions
   * @param x_min, x_max X domain bounds
   * @param y_min, y_max Y domain bounds
   * @return Generated 2D mesh
   */
  static std::unique_ptr<MeshBase> build_cartesian_2d(
      int nx, int ny,
      real_t x_min, real_t x_max,
      real_t y_min, real_t y_max);

  /**
   * @brief Build 3D Cartesian grid with custom spacing
   * @param x_coords X coordinates of grid points
   * @param y_coords Y coordinates of grid points
   * @param z_coords Z coordinates of grid points
   * @return Generated mesh
   */
  static std::unique_ptr<MeshBase> build_cartesian_custom(
      const std::vector<real_t>& x_coords,
      const std::vector<real_t>& y_coords,
      const std::vector<real_t>& z_coords);

  /**
   * @brief Build cylindrical mesh
   * @param nr, ntheta, nz Number of cells in r, theta, z
   * @param r_min, r_max Radial bounds
   * @param z_min, z_max Axial bounds
   * @param theta_min, theta_max Angular bounds (default 0 to 2*pi)
   * @return Generated cylindrical mesh
   */
  static std::unique_ptr<MeshBase> build_cylindrical(
      int nr, int ntheta, int nz,
      real_t r_min, real_t r_max,
      real_t z_min, real_t z_max,
      real_t theta_min = 0, real_t theta_max = 2*M_PI);

  /**
   * @brief Build spherical mesh
   * @param nr, ntheta, nphi Number of cells
   * @param r_min, r_max Radial bounds
   * @param theta_min, theta_max Polar angle bounds
   * @param phi_min, phi_max Azimuthal angle bounds
   * @return Generated spherical mesh
   */
  static std::unique_ptr<MeshBase> build_spherical(
      int nr, int ntheta, int nphi,
      real_t r_min, real_t r_max,
      real_t theta_min = 0, real_t theta_max = M_PI,
      real_t phi_min = 0, real_t phi_max = 2*M_PI);

  // ---- Mesh extrusion ----

  /**
   * @brief Extrude 2D mesh to 3D
   * @param base_2d 2D base mesh
   * @param n_layers Number of layers
   * @param height Total extrusion height
   * @return Extruded 3D mesh
   */
  static std::unique_ptr<MeshBase> extrude(
      const MeshBase& base_2d,
      int n_layers,
      real_t height);

  /**
   * @brief Extrude with variable layer heights
   * @param base_2d 2D base mesh
   * @param layer_heights Height of each layer
   * @return Extruded 3D mesh
   */
  static std::unique_ptr<MeshBase> extrude_variable(
      const MeshBase& base_2d,
      const std::vector<real_t>& layer_heights);

  /**
   * @brief Extrude along path
   * @param base_2d 2D cross-section mesh
   * @param path Path points for extrusion
   * @param n_segments Number of segments along path
   * @return Extruded mesh along path
   */
  static std::unique_ptr<MeshBase> extrude_along_path(
      const MeshBase& base_2d,
      const std::vector<std::array<real_t,3>>& path,
      int n_segments);

  /**
   * @brief Revolve 2D mesh around axis
   * @param base_2d 2D profile mesh
   * @param n_theta Number of angular divisions
   * @param angle Total revolution angle (default 2*pi)
   * @param axis Axis of revolution (0=x, 1=y, 2=z)
   * @return Revolved 3D mesh
   */
  static std::unique_ptr<MeshBase> revolve(
      const MeshBase& base_2d,
      int n_theta,
      real_t angle = 2*M_PI,
      int axis = 2);

  // ---- Primitive shapes ----

  /**
   * @brief Build box mesh
   * @param nx, ny, nz Divisions in each direction
   * @param center Box center
   * @param dimensions Box dimensions (width, height, depth)
   * @return Box mesh
   */
  static std::unique_ptr<MeshBase> build_box(
      int nx, int ny, int nz,
      const std::array<real_t,3>& center = {{0,0,0}},
      const std::array<real_t,3>& dimensions = {{1,1,1}});

  /**
   * @brief Build sphere mesh
   * @param n_subdivisions Number of subdivision levels
   * @param radius Sphere radius
   * @param center Sphere center
   * @param method Generation method (ico, uv, cube)
   * @return Sphere mesh
   */
  static std::unique_ptr<MeshBase> build_sphere(
      int n_subdivisions,
      real_t radius = 1.0,
      const std::array<real_t,3>& center = {{0,0,0}},
      const std::string& method = "ico");

  /**
   * @brief Build cylinder mesh
   * @param nr, ntheta, nz Divisions
   * @param radius Cylinder radius
   * @param height Cylinder height
   * @param center Cylinder center
   * @return Cylinder mesh
   */
  static std::unique_ptr<MeshBase> build_cylinder(
      int nr, int ntheta, int nz,
      real_t radius = 1.0,
      real_t height = 1.0,
      const std::array<real_t,3>& center = {{0,0,0}});

  /**
   * @brief Build torus mesh
   * @param n_major Major radius divisions
   * @param n_minor Minor radius divisions
   * @param major_radius Major radius
   * @param minor_radius Minor radius
   * @param center Torus center
   * @return Torus mesh
   */
  static std::unique_ptr<MeshBase> build_torus(
      int n_major, int n_minor,
      real_t major_radius = 1.0,
      real_t minor_radius = 0.3,
      const std::array<real_t,3>& center = {{0,0,0}});

  // ---- Mesh combination ----

  /**
   * @brief Merge multiple meshes
   * @param meshes Vector of meshes to merge
   * @param tolerance Tolerance for duplicate node removal
   * @return Merged mesh
   */
  static std::unique_ptr<MeshBase> merge(
      const std::vector<const MeshBase*>& meshes,
      real_t tolerance = 1e-10);

  /**
   * @brief Append mesh (no duplicate removal)
   * @param mesh1 First mesh
   * @param mesh2 Second mesh
   * @return Combined mesh
   */
  static std::unique_ptr<MeshBase> append(
      const MeshBase& mesh1,
      const MeshBase& mesh2);

  /**
   * @brief Boolean union of meshes
   * @param mesh1 First mesh
   * @param mesh2 Second mesh
   * @return Union mesh
   */
  static std::unique_ptr<MeshBase> boolean_union(
      const MeshBase& mesh1,
      const MeshBase& mesh2);

  /**
   * @brief Boolean intersection of meshes
   * @param mesh1 First mesh
   * @param mesh2 Second mesh
   * @return Intersection mesh
   */
  static std::unique_ptr<MeshBase> boolean_intersection(
      const MeshBase& mesh1,
      const MeshBase& mesh2);

  /**
   * @brief Boolean difference of meshes
   * @param mesh1 First mesh
   * @param mesh2 Second mesh
   * @return Difference mesh (mesh1 - mesh2)
   */
  static std::unique_ptr<MeshBase> boolean_difference(
      const MeshBase& mesh1,
      const MeshBase& mesh2);

  // ---- Mesh transformation ----

  /**
   * @brief Transform mesh coordinates
   * @param mesh Input mesh
   * @param transform Transformation function
   * @return Transformed mesh
   */
  static std::unique_ptr<MeshBase> transform(
      const MeshBase& mesh,
      std::function<std::array<real_t,3>(const std::array<real_t,3>&)> transform);

  /**
   * @brief Translate mesh
   * @param mesh Input mesh
   * @param translation Translation vector
   * @return Translated mesh
   */
  static std::unique_ptr<MeshBase> translate(
      const MeshBase& mesh,
      const std::array<real_t,3>& translation);

  /**
   * @brief Rotate mesh
   * @param mesh Input mesh
   * @param axis Rotation axis
   * @param angle Rotation angle in radians
   * @param center Rotation center
   * @return Rotated mesh
   */
  static std::unique_ptr<MeshBase> rotate(
      const MeshBase& mesh,
      const std::array<real_t,3>& axis,
      real_t angle,
      const std::array<real_t,3>& center = {{0,0,0}});

  /**
   * @brief Scale mesh
   * @param mesh Input mesh
   * @param scale Scale factors for each axis
   * @param center Scaling center
   * @return Scaled mesh
   */
  static std::unique_ptr<MeshBase> scale(
      const MeshBase& mesh,
      const std::array<real_t,3>& scale,
      const std::array<real_t,3>& center = {{0,0,0}});

  // ---- Mesh from point cloud ----

  /**
   * @brief Create mesh from point cloud using Delaunay
   * @param points Point cloud
   * @param dimension Spatial dimension (2 or 3)
   * @return Delaunay mesh
   */
  static std::unique_ptr<MeshBase> delaunay(
      const std::vector<std::array<real_t,3>>& points,
      int dimension = 3);

  /**
   * @brief Create mesh from point cloud using Voronoi
   * @param points Point cloud
   * @param dimension Spatial dimension (2 or 3)
   * @return Voronoi mesh
   */
  static std::unique_ptr<MeshBase> voronoi(
      const std::vector<std::array<real_t,3>>& points,
      int dimension = 3);

  // ---- Mesh refinement ----

  /**
   * @brief Uniform refinement
   * @param mesh Input mesh
   * @param levels Number of refinement levels
   * @return Refined mesh
   */
  static std::unique_ptr<MeshBase> refine_uniform(
      const MeshBase& mesh,
      int levels = 1);

  /**
   * @brief Adaptive refinement based on field
   * @param mesh Input mesh
   * @param field_handle Field to guide refinement
   * @param threshold Refinement threshold
   * @return Refined mesh
   */
  static std::unique_ptr<MeshBase> refine_adaptive(
      const MeshBase& mesh,
      const FieldHandle& field_handle,
      real_t threshold);

private:
  // Helper methods for structured grid generation
  static void create_hex_connectivity(
      std::vector<index_t>& connectivity,
      std::vector<offset_t>& offsets,
      std::vector<CellShape>& shapes,
      int nx, int ny, int nz);

  static void create_quad_connectivity(
      std::vector<index_t>& connectivity,
      std::vector<offset_t>& offsets,
      std::vector<CellShape>& shapes,
      int nx, int ny);
};

} // namespace svmp

#endif // SVMP_MESH_BUILDERS_H