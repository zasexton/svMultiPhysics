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

#ifndef SVMP_POLYHEDRON_TESSELLATION_H
#define SVMP_POLYHEDRON_TESSELLATION_H

#include "../Core/MeshTypes.h"
#include <array>
#include <cstddef>
#include <vector>

namespace svmp {

class MeshBase;

struct PolyhedronTet4 {
  std::array<std::array<real_t, 3>, 4> vertices{};
};

/**
 * @brief Helper utilities for tetrahedralizing polyhedron cells.
 *
 * Current implementation provides a convex "star" tessellation using an interior point
 * (vertex-average) and a fan triangulation of each face polygon.
 *
 * Note: This is intended for quality evaluation / visualization and assumes faces are
 * planar polygons and the polyhedron is convex.
 */
class PolyhedronTessellation {
public:
  /**
   * @brief Return deterministic linear tetrahedra for a linear volume cell.
   *
   * This centralizes the canonical tetrahedral decompositions used by cut
   * topology, measure estimation, and validation code. Tetra, hex, wedge, and
   * pyramid cells are decomposed from their standard corner ordering. Polyhedron
   * cells use the convex star tessellation below.
   */
  static std::vector<PolyhedronTet4> linear_cell_tets(
      const MeshBase& mesh,
      index_t cell,
      Configuration cfg = Configuration::Reference);

  /**
   * @brief Return deterministic linear tetrahedra for explicit linear-cell points.
   *
   * The input points must follow the canonical corner ordering for the requested
   * family. This overload is used by tessellated high-order cells, where the
   * subcell points are already physical coordinates rather than mesh vertices.
   */
  static std::vector<PolyhedronTet4> linear_cell_tets(
      CellFamily family,
      const std::vector<std::array<real_t, 3>>& points);

  /**
   * @brief Return the canonical corner-index tetrahedra used for a linear volume cell.
   *
   * The returned indices are the same deterministic decompositions used by
   * linear_cell_tets(). Consumers that need to carry parallel per-corner data
   * such as parent-parametric coordinates can use this contract without
   * duplicating the decomposition tables.
   */
  static std::vector<std::array<std::size_t, 4>> linear_cell_tet_corner_indices(
      CellFamily family,
      std::size_t point_count);

  static std::vector<PolyhedronTet4> convex_star_tets(
      const MeshBase& mesh,
      index_t cell,
      Configuration cfg = Configuration::Reference);
};

} // namespace svmp

#endif // SVMP_POLYHEDRON_TESSELLATION_H
