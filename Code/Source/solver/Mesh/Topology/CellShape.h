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

#ifndef SVMP_CELL_SHAPE_H
#define SVMP_CELL_SHAPE_H

#include "../Core/MeshTypes.h"
#include <unordered_map>
#include <string>
#include <vector>

namespace svmp {

// Uses EntityKind (Vertex/Line/Face/Volume) to represent topological kind

// --------------------
// Cell shape metadata
// --------------------
struct CellShape {
  // Compatibility constants to allow usages like CellShape::Hexahedron
  static constexpr CellFamily Line         = CellFamily::Line;
  static constexpr CellFamily Triangle     = CellFamily::Triangle;
  static constexpr CellFamily Quad         = CellFamily::Quad;
  static constexpr CellFamily Quadrilateral= CellFamily::Quad;
  static constexpr CellFamily Tetrahedron  = CellFamily::Tetra;
  static constexpr CellFamily Hexahedron   = CellFamily::Hex;
  static constexpr CellFamily Wedge        = CellFamily::Wedge;
  static constexpr CellFamily Prism        = CellFamily::Wedge;
  static constexpr CellFamily Pyramid      = CellFamily::Pyramid;
  CellFamily family = CellFamily::Polygon;
  int num_corners = 0;        // number of corner vertices (for poly: >= 3)
  int order = 1;              // geometric/approximation order
  bool is_mixed_order = false;
  // Optional hints for variable-topology families (e.g., polyhedra & polygons)
  int num_faces_hint = -1;    // number of faces for Polyhedron when known
  int num_edges_hint = -1;    // number of edges for Polyhedron/Polygon when known

  // Helper methods
  bool is_linear() const { return order == 1; }
  bool is_quadratic() const { return order == 2; }
  bool is_high_order() const { return order > 2; }
  bool is_1d() const { return family == CellFamily::Line; }
  bool is_2d() const {
    return family == CellFamily::Triangle ||
           family == CellFamily::Quad ||
           family == CellFamily::Polygon;
  }
  bool is_3d() const {
    return family == CellFamily::Tetra ||
           family == CellFamily::Hex ||
           family == CellFamily::Wedge ||
           family == CellFamily::Pyramid ||
           family == CellFamily::Polyhedron;
  }

  // Topological kind derived from cell family (by dimension)
  EntityKind topo_kind() const;  // Implemented after CellShapeUtils

  // Expected vertex count for standard elements
  //
  // Convention
  // - Returns the Lagrange (complete) node count by default for order >= 1.
  // - Some formats also use Serendipity variants with fewer interior nodes.
  //   Where these are common we mention them explicitly below, but this
  //   function still returns the Lagrange count to keep high‑order logic
  //   consistent with complete polynomial spaces.
  int expected_vertices() const {
    if (family == CellFamily::Point) {
      return 1;
    } else if (family == CellFamily::Line) {
      // Lagrange: p+1 nodes; Serendipity is not applicable for 1D
      return order + 1;
    } else if (family == CellFamily::Triangle) {
      // Lagrange (standard): p=1 -> 3, p=2 -> 6, p>=3 -> (p+1)(p+2)/2
      // Serendipity: uncommon for triangles (Lagrange is typical)
      if (order == 1) return 3;
      if (order == 2) return 6;
      return (order + 1) * (order + 2) / 2;
    } else if (family == CellFamily::Quad) {
      // Lagrange: p=1 -> 4, p=2 -> 9, p>=3 -> (p+1)^2
      // Serendipity (common): p=2 -> 8 (Q8), p=3 -> 12 (Q12), in general 4 + 4*(p-1)
      if (order == 1) return 4;
      if (order == 2) return 9;
      return (order + 1) * (order + 1);
    } else if (family == CellFamily::Tetra) {
      // Lagrange: p=1 -> 4, p=2 -> 10, p>=3 -> (p+1)(p+2)(p+3)/6
      // Serendipity: rare for tetrahedra; Lagrange is typical
      if (order == 1) return 4;
      if (order == 2) return 10;
      return (order + 1) * (order + 2) * (order + 3) / 6;
    } else if (family == CellFamily::Hex) {
      // Lagrange: p=1 -> 8, p=2 -> 27, p>=3 -> (p+1)^3
      // Serendipity (common): p=2 -> 20 (HEX20), higher p often omit interior nodes
      if (order == 1) return 8;
      if (order == 2) return 27;
      return (order + 1) * (order + 1) * (order + 1);
    } else if (family == CellFamily::Wedge) {
      // Lagrange: p=1 -> 6, p=2 -> 18, p>=3 -> (p+1)(p+1)(p+2)/2
      // Serendipity (VTK quadratic wedge): 15 nodes (edge midpoints only)
      if (order == 1) return 6;
      if (order == 2) return 18;
      return (order + 1) * (order + 1) * (order + 2) / 2;
    } else if (family == CellFamily::Pyramid) {
      // Lagrange pyramid total nodes (VTK): N = sum_{m=1}^{p+1} m^2 = (p+1)(p+2)(2p+3)/6
      // Legacy quadratic pyramids often appear with 13 (no base center) or 14 nodes; this
      // function returns the Lagrange count (14 for p=2).
      return (order + 1) * (order + 2) * (2 * order + 3) / 6;
    }
    // Polygon/Polyhedron: variable vertex count
    return -1;
  }
};

// --------------------
// Format-agnostic cell shape registry
// --------------------
// IO modules can register mappings between their format's cell type IDs and CellShape
class CellShapeRegistry {
public:
  // Register a mapping from format-specific type ID to CellShape
  static void register_shape(const std::string& format, int type_id, const CellShape& shape);

  // Check if a mapping exists
  static bool has(const std::string& format, int type_id);

  // Get the CellShape for a format-specific type ID
  static CellShape get(const std::string& format, int type_id);

  // Clear all mappings for a format
  static void clear_format(const std::string& format);

  // Clear all mappings
  static void clear_all();

  // Get all registered formats
  static std::vector<std::string> formats();

private:
  // Internal storage for format -> (type_id -> CellShape) mappings
  static std::unordered_map<std::string, std::unordered_map<int, CellShape>>& map_();
};

// --------------------
// Common cell shape utilities
// --------------------
namespace CellShapeUtils {
  // Get the number of faces for a standard cell type
  inline int num_faces(const CellShape& shape) {
    switch (shape.family) {
      case CellFamily::Line: return 0;       // 1 face
      case CellFamily::Triangle: return 1;   // 1 face
      case CellFamily::Quad: return 1;       // 1 face
      case CellFamily::Polygon: return 1;    // 1 face
      case CellFamily::Tetra: return 4;      // 4 triangular faces
      case CellFamily::Hex: return 6;        // 6 quadrilateral faces
      case CellFamily::Wedge: return 5;      // 2 triangles + 3 quads
      case CellFamily::Pyramid: return 5;    // 1 quad + 4 triangles
      default: return -1;                    // Variable for poly
    }
  }

  // Get the number of edges for a standard cell type
  inline int num_edges(const CellShape& shape) {
    switch (shape.family) {
      case CellFamily::Line: return 1;       // The line itself
      case CellFamily::Triangle: return 3;
      case CellFamily::Quad: return 4;
      case CellFamily::Tetra: return 6;
      case CellFamily::Hex: return 12;
      case CellFamily::Wedge: return 9;
      case CellFamily::Pyramid: return 8;
      default: return -1;                    // Variable for polygon & polyhedron
    }
  }

  // Get the topological entity kind for a cell family (by dimension)
  inline EntityKind dimension(CellFamily family) {
    switch (family) {
      case CellFamily::Line:
        return EntityKind::Edge;     // 1D
      case CellFamily::Triangle:
      case CellFamily::Quad:
      case CellFamily::Polygon:
        return EntityKind::Face;     // 2D
      case CellFamily::Tetra:
      case CellFamily::Hex:
      case CellFamily::Wedge:
      case CellFamily::Pyramid:
      case CellFamily::Polyhedron:
        return EntityKind::Volume;   // 3D
    }
    return EntityKind::Vertex;       // default/fallback (0D)
  }
}

// Implementation of CellShape::topo_kind() after CellShapeUtils is defined
inline EntityKind CellShape::topo_kind() const {
  return CellShapeUtils::dimension(family);
}

} // namespace svmp

#endif // SVMP_CELL_SHAPE_H
