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

#ifndef SVMP_MESH_TYPES_H
#define SVMP_MESH_TYPES_H

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace svmp {

// ------------------------
// Fundamental type aliases
// ------------------------
using index_t = int32_t;      // local indices (vertices/faces/cells)
using offset_t = int64_t;     // CSR offsets (can handle > 2B entries)
using gid_t   = int64_t;      // global IDs across MPI ranks
using real_t  = double;       // geometry coordinates
using rank_t  = int32_t;      // MPI rank
using label_t = int32_t;      // region / boundary / material tags

// ---------
// Constants
// ---------
constexpr index_t INVALID_INDEX = -1;
constexpr label_t INVALID_LABEL = -1;

// ---------
// Enums
// ---------
enum class EntityKind {
  Vertex = 0,
  Edge   = 1,
  Face   = 2,
  Volume = 3   // 3D cells
};

enum class Ownership {
  Owned,
  Ghost,
  Shared
};

enum class MappingKind {
  Affine,
  Isoparametric,
  Curvilinear,
  NURBS
};

enum class Configuration {
  Reference,
  Current
};

/**
 * @brief Topological and geometric constraint types
 *
 * Constraint types that represent special topological/geometric relationships
 * between mesh entities. These are mesh-level constraints only - physics-specific
 * constraints (e.g., contact, friction) belong in the Solver, not here.
 */
enum class ConstraintKind {
  None,      // No constraint
  Hanging,   // Hanging vertex from adaptive refinement
  Periodic,  // Periodic boundary equivalence
  Mortar     // Non-conforming interface topology
};

enum class CellFamily {
  Point,
  Line,
  Triangle,
  Quad,
  Tetra,
  Hex,
  Wedge,
  Pyramid,
  Polygon,
  Polyhedron
};

enum class ReorderAlgo {
  None,
  RCM,
  CuthillMcKee,
  Hilbert,
  Morton
};

enum class PartitionHint {
  None,
  Cells,      // Balance by number of cells
  Vertices,   // Balance by number of vertices
  Memory,     // Balance by memory usage
  Metis,      // Use METIS graph partitioning
  ParMetis,
  Scotch,
  Zoltan,
  Custom
};

// Scalar type for field attachments (extend as needed)
enum class FieldScalarType {
  Int32,
  Int64,
  Float32,
  Float64,
  UInt8,
  Custom
};

inline constexpr size_t bytes_per(FieldScalarType t) noexcept {
  switch (t) {
    case FieldScalarType::Int32:   return 4;
    case FieldScalarType::Int64:   return 8;
    case FieldScalarType::Float32: return 4;
    case FieldScalarType::Float64: return 8;
    case FieldScalarType::UInt8:   return 1;
    case FieldScalarType::Custom:  return 0;
  }
  return 0;
}

// --------------------
// Basic data structures
// --------------------
struct BoundingBox {
  std::array<real_t,3> min { {+1e300, +1e300, +1e300} };
  std::array<real_t,3> max { {-1e300, -1e300, -1e300} };
};

struct FieldHandle {
  uint32_t id = 0;
  EntityKind kind = EntityKind::Vertex;
  std::string name;
};

// ------------------
// Mesh IO structures
// ------------------
struct MeshIOOptions {
  std::string format;                                     // "vtu", "gmsh", "exodus", ...
  std::string path;                                       // file path
  std::unordered_map<std::string,std::string> kv;        // extra options
};

// --------------------
// Search result structures
// --------------------
struct PointLocateResult {
  index_t cell_id = -1;
  std::array<real_t,3> xi = {{0,0,0}};
  bool found = false;
};

struct RayIntersectResult {
  index_t face_id = -1;
  real_t t = -1.0;
  std::array<real_t,3> point = {{0,0,0}};
  bool found = false;
};

} // namespace svmp

#endif // SVMP_MESH_TYPES_H
