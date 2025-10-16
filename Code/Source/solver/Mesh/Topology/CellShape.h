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
  CellFamily family = CellFamily::Polygon;
  int num_corners = 0;        // number of corner nodes (for poly: >= 3)
  int order = 1;              // geometric/approximation order
  bool is_mixed_order = false;
  // Optional hints for variable-topology families (e.g., polyhedra & polygons)
  int num_faces_hint = -1;    // number of faces for Polyhedron when known
  int num_edges_hint = -1;    // number of edges for Polyhedron/Polygon when known

  // Helper methods
  bool is_linear() const { return order == 1; }
  bool is_quadratic() const { return order == 2; }
  bool is_high_order() const { return order > 2; }
  bool is_1d() const { return family == CellFamily::Point; }
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

  // Expected node count for standard elements
  int expected_nodes() const {
    if (family == CellFamily::Point) {
      return 1;
    } else if (family == CellFamily::Line) {
      return order + 1;
    } else if (family == CellFamily::Triangle) {
      if (order == 1) return 3;
      if (order == 2) return 6;
      return (order + 1) * (order + 2) / 2;
    } else if (family == CellFamily::Quad) {
      if (order == 1) return 4;
      if (order == 2) return 9;
      return (order + 1) * (order + 1);
    } else if (family == CellFamily::Tetra) {
      if (order == 1) return 4;
      if (order == 2) return 10;
      return (order + 1) * (order + 2) * (order + 3) / 6;
    } else if (family == CellFamily::Hex) {
      if (order == 1) return 8;
      if (order == 2) return 27;
      return (order + 1) * (order + 1) * (order + 1);
    } else if (family == CellFamily::Wedge) {
      if (order == 1) return 6;
      if (order == 2) return 18;
      return (order + 1) * (order + 1) * (order + 2) / 2;
    } else if (family == CellFamily::Pyramid) {
      if (order == 1) return 5;
      if (order == 2) return 14;
      // Pyramids have irregular node counts for high order
      return 5 + 8 * (order - 1) + (order - 1) * (order - 2) / 2;
    }
    // Polygon/Polyhedron: variable node count
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
