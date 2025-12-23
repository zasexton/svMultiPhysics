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

#ifndef SVMP_POLY_GEOMETRY_H
#define SVMP_POLY_GEOMETRY_H

#include "../Core/MeshBase.h"
#include <array>
#include <vector>

namespace svmp {

/**
 * @brief Robust polygon and (convex) polyhedron geometry utilities.
 *
 * These helpers operate on either explicit 3D vertex arrays or on
 * a MeshBase with a list of vertex indices. For mesh-based variants,
 * coordinates are taken from the selected configuration (Reference or Current).
 *
 * Polyhedron routines require explicit face connectivity on the mesh
 * (i.e., `MeshBase::face_vertices_span` and `MeshBase::cell_faces` must be valid).
 * Current implementations assume a closed, watertight, convex polyhedron.
 */
class PolyGeometry {
public:
  struct PolyhedronMassProperties {
    real_t volume = 0.0;               // positive volume
    std::array<real_t,3> centroid{};   // geometric centroid
    bool is_valid = false;            // false if faces are missing/non-manifold/inconsistent
  };

  // ---- 3D coordinates API ----

  // Newell normal (unnormalized). Returns zero vector for degenerate input.
  static std::array<real_t,3> newell_normal(const std::vector<std::array<real_t,3>>& verts);

  // Polygon area using Newell's method (0 for degenerate or < 3 vertices).
  static real_t polygon_area(const std::vector<std::array<real_t,3>>& verts);

  // Polygon centroid using triangle fan decomposition (robust for planar polygon).
  static std::array<real_t,3> polygon_centroid(const std::vector<std::array<real_t,3>>& verts);

  // ---- Mesh-based API (indices into mesh vertex list) ----

  static std::array<real_t,3> polygon_normal(const MeshBase& mesh,
                                             const std::vector<index_t>& vertices,
                                             Configuration cfg = Configuration::Reference);

  static real_t polygon_area(const MeshBase& mesh,
                             const std::vector<index_t>& vertices,
                             Configuration cfg = Configuration::Reference);

  static std::array<real_t,3> polygon_centroid(const MeshBase& mesh,
                                               const std::vector<index_t>& vertices,
                                               Configuration cfg = Configuration::Reference);

  // ---- Convex polyhedra (mesh-based; face connectivity required) ----

  // Polyhedron volume via decomposition into tetrahedra from an interior point (convex only).
  static real_t polyhedron_volume(const MeshBase& mesh,
                                  index_t cell,
                                  Configuration cfg = Configuration::Reference);

  // Polyhedron centroid via tetrahedral decomposition from an interior point (convex only).
  static std::array<real_t,3> polyhedron_centroid(const MeshBase& mesh,
                                                  index_t cell,
                                                  Configuration cfg = Configuration::Reference);

  // General (possibly non-convex) polyhedron mass properties using oriented faces and
  // a divergence-theorem surface integral (requires a watertight, manifold face set).
  static PolyhedronMassProperties polyhedron_mass_properties(const MeshBase& mesh,
                                                             index_t cell,
                                                             Configuration cfg = Configuration::Reference);
};

} // namespace svmp

#endif // SVMP_POLY_GEOMETRY_H
