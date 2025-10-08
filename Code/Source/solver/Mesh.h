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

#ifndef SVMP_MESH_H
#define SVMP_MESH_H

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svmp {

// ------------------------
// Fundamental type aliases
// ------------------------
using index_t = int32_t;      // local indices (nodes/faces/cells)
using gid_t   = int64_t;      // global IDs across MPI ranks
using real_t  = double;       // geometry coordinates
using rank_t  = int32_t;      // MPI rank
using label_t = int32_t;      // region / boundary / material tags

// ---------
// Enums
// ---------
enum class EntityKind { Vertex = 0, Edge = 1, Face = 2, Cell = 3 };
enum class Ownership { Owned, Ghost, Shared };
enum class MappingKind { Affine, Isoparametric, Curvilinear, NURBS };
enum class Configuration { Reference, Current };
enum class ConstraintKind { None, Hanging, Periodic, Mortar, Contact };
enum class CellFamily { Line, Triangle, Quad, Tetra, Hex, Wedge, Pyramid, Polygon, Polyhedron };
enum class ReorderAlgo { None, RCM, CuthillMcKee, Hilbert, Morton };
enum class PartitionHint { None, Metis, ParMetis, Scotch, Zoltan, Custom };

// Scalar type for field attachments (extend as needed)
enum class FieldScalarType { Int32, Int64, Float32, Float64, UInt8, Custom };

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
// Cell shape metadata
// --------------------
struct CellShape {
  int vtk_type_id = -1;       // VTK cell type id if applicable
  CellFamily family = CellFamily::Polygon;
  int num_corners = 0;        // number of corner nodes (for poly: >= 3)
  int order = 1;              // geometric/approximation order
  bool is_mixed_order = false;
};

// Minimal registry for cell shapes, extensible to cover all VTK types
class CellShapeRegistry {
public:
  static void register_shape(int vtk_id, const CellShape& shape);
  static bool has(int vtk_id);
  static CellShape get(int vtk_id);
  static void register_default_vtk_core(); // Tri3/Quad4/Tet4/Hex8 minimal set

private:
  static std::unordered_map<int, CellShape>& map_();
};

// ------------------
// Mesh IO interfaces
// ------------------
struct MeshIOOptions {
  std::string format;                   // "vtu", "gmsh", "exodus", ...
  std::string path;                     // file path
  std::unordered_map<std::string,std::string> kv; // extra options
};

// ----------------
// Mesh core class
// ----------------
// Runtime-dimensional base container with modular, extensible design.
class MeshBase {
public:
  // ---- Types
  struct BoundingBox {
    std::array<real_t,3> min { {+1e300, +1e300, +1e300} };
    std::array<real_t,3> max { {-1e300, -1e300, -1e300} };
  };

  struct FieldHandle { uint32_t id = 0; EntityKind kind = EntityKind::Vertex; std::string name; };

  // ---- Lifecycle
  MeshBase();
  explicit MeshBase(int spatial_dim);

  // Builders
  void clear();
  void reserve(index_t n_nodes, index_t n_cells, index_t n_faces = 0);
  void build_from_arrays(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<index_t>& cell2node_offsets,
      const std::vector<index_t>& cell2node,
      const std::vector<CellShape>& cell_shape);
  void set_faces_from_arrays(
      const std::vector<CellShape>& face_shape,
      const std::vector<index_t>& face2node_offsets,
      const std::vector<index_t>& face2node,
      const std::vector<std::array<index_t,2>>& face2cell);
  void finalize(); // build minimal caches, validate basic invariants

  // ---- Basic queries
  int dim() const noexcept { return spatial_dim_; }
  size_t n_nodes()  const noexcept { return X_ref_.size() / (spatial_dim_ > 0 ? spatial_dim_ : 1); }
  size_t n_cells()  const noexcept { return cell_shape_.size(); }
  size_t n_faces()  const noexcept { return face_shape_.size(); }
  size_t n_edges()  const noexcept { return 0; } // optional for now

  // Coordinates (reference/current)
  const std::vector<real_t>& X_ref() const noexcept { return X_ref_; }
  const std::vector<real_t>& X_cur() const noexcept { return X_cur_; }
  bool has_current_coords() const noexcept { return !X_cur_.empty(); }
  void set_current_coords(const std::vector<real_t>& Xcur);
  void clear_current_coords();

  // Topology access
  const std::vector<CellShape>& cell_shapes() const noexcept { return cell_shape_; }
  const CellShape& cell_shape(index_t c) const { return cell_shape_.at(static_cast<size_t>(c)); }
  std::pair<const index_t*, size_t> cell_nodes_span(index_t c) const;
  const std::vector<index_t>& cell2node_offsets() const noexcept { return cell2node_offsets_; }
  const std::vector<index_t>& cell2node() const noexcept { return cell2node_; }

  // Faces (optional if not provided by builder)
  const std::vector<CellShape>& face_shapes() const noexcept { return face_shape_; }
  std::pair<const index_t*, size_t> face_nodes_span(index_t f) const;
  const std::array<index_t,2>& face_cells(index_t f) const { return face2cell_.at(static_cast<size_t>(f)); }

  // IDs / ownership
  const std::vector<gid_t>& node_gids() const noexcept { return node_gid_; }
  const std::vector<gid_t>& cell_gids() const noexcept { return cell_gid_; }
  const std::vector<gid_t>& face_gids() const noexcept { return face_gid_; }
  void set_node_gids(std::vector<gid_t> gids) { node_gid_ = std::move(gids); }
  void set_cell_gids(std::vector<gid_t> gids) { cell_gid_ = std::move(gids); }
  void set_face_gids(std::vector<gid_t> gids) { face_gid_ = std::move(gids); }

  // Labels / sets
  void set_region_label(index_t cell, label_t label);
  label_t region_label(index_t cell) const;
  std::vector<index_t> cells_with_label(label_t label) const;

  void set_boundary_label(index_t face, label_t label);
  label_t boundary_label(index_t face) const;
  std::vector<index_t> faces_with_label(label_t label) const;

  void add_to_set(EntityKind kind, const std::string& name, index_t id);
  const std::vector<index_t>& get_set(EntityKind kind, const std::string& name) const;
  bool has_set(EntityKind kind, const std::string& name) const;

  // Attachment system (fields on entities)
  FieldHandle attach_field(EntityKind kind, const std::string& name, FieldScalarType type, size_t components);
  bool has_field(EntityKind kind, const std::string& name) const;
  void remove_field(const FieldHandle& h);
  void* field_data(const FieldHandle& h);
  const void* field_data(const FieldHandle& h) const;
  size_t field_components(const FieldHandle& h) const;
  FieldScalarType field_type(const FieldHandle& h) const;
  size_t field_entity_count(const FieldHandle& h) const;

  template <typename T>
  T* field_data_as(const FieldHandle& h) {
    return reinterpret_cast<T*>(field_data(h));
  }
  template <typename T>
  const T* field_data_as(const FieldHandle& h) const {
    return reinterpret_cast<const T*>(field_data(h));
  }

  // Geometry helpers (centroid, measure; measure optional for complex cells)
  std::array<real_t,3> cell_center(index_t c, Configuration cfg = Configuration::Reference) const;
  BoundingBox bounding_box(Configuration cfg = Configuration::Reference) const;

  // Reordering (no-op default; hooks for future algorithms)
  void reorder(ReorderAlgo algo);

  // Validation & diagnostics (lightweight)
  void validate_basic() const; // sizes and offsets sanity

  // IO registry (format-agnostic)
  using LoadFn = std::function<MeshBase(const MeshIOOptions&)>;
  using SaveFn = std::function<void(const MeshBase&, const MeshIOOptions&)>;
  static void register_reader(const std::string& format, LoadFn fn);
  static void register_writer(const std::string& format, SaveFn fn);
  static MeshBase load(const MeshIOOptions& opts);
  void save(const MeshIOOptions& opts) const;
  static std::vector<std::string> registered_readers();
  static std::vector<std::string> registered_writers();

private:
  // Helpers
  size_t entity_count(EntityKind k) const noexcept;
  void invalidate_caches();

private:
  int spatial_dim_ = 0;

  // Geometry
  std::vector<real_t> X_ref_;  // size = n_nodes * spatial_dim
  std::vector<real_t> X_cur_;  // optional current (ALE/FSI) same size as X_ref

  // Topology: cells
  std::vector<CellShape>  cell_shape_;
  std::vector<index_t>    cell2node_offsets_; // CSR offsets size n_cells+1
  std::vector<index_t>    cell2node_;         // concatenated node indices

  // Topology: faces (optional)
  std::vector<CellShape>  face_shape_;
  std::vector<index_t>    face2node_offsets_;
  std::vector<index_t>    face2node_;
  std::vector<std::array<index_t,2>> face2cell_; // inner, outer(-1 for boundary)

  // IDs / ownership
  std::vector<gid_t>      node_gid_;
  std::vector<gid_t>      face_gid_;
  std::vector<gid_t>      cell_gid_;
  std::vector<Ownership>  node_owner_;
  std::vector<Ownership>  face_owner_;
  std::vector<Ownership>  cell_owner_;

  // Labels / sets
  std::vector<label_t> cell_region_id_;
  std::vector<label_t> face_boundary_id_;
  std::unordered_map<std::string, std::vector<index_t>> entity_sets_[4];

  // Attachments: type-erased field storage per entity kind
  struct FieldInfo { FieldScalarType type; size_t components; std::vector<uint8_t> data; };
  struct AttachTable { std::unordered_map<std::string, FieldInfo> by_name; };
  AttachTable attachments_[4];
  uint32_t next_field_id_ = 1;
  std::unordered_map<uint32_t, std::pair<EntityKind, std::string>> field_index_; // id -> (kind,name)

  // Adaptivity (placeholders)
  std::vector<index_t> cell_parent_;                  // parent cell id or -1 if root
  std::vector<index_t> cell_children_offsets_;        // CSR offsets into cell_children_
  std::vector<index_t> cell_children_;                // concatenated children ids
  std::vector<uint8_t> refine_flag_;                  // 1=refine, 0=keep
  std::vector<uint8_t> coarsen_flag_;                 // 1=coarsen, 0=keep

  // IO registry
  static std::unordered_map<std::string, LoadFn>& readers_();
  static std::unordered_map<std::string, SaveFn>& writers_();
};

// -------------------------------
// Compile-time typed mesh view
// -------------------------------
template <int Dim>
class Mesh {
public:
  explicit Mesh(std::shared_ptr<MeshBase> base)
    : base_(std::move(base))
  {
    if (!base_) throw std::invalid_argument("Mesh<Dim>: null base");
    if (base_->dim() != 0 && base_->dim() != Dim) {
      throw std::invalid_argument("Mesh<Dim>: base dim mismatch");
    }
  }

  int dim() const noexcept { return Dim; }
  const MeshBase& base() const noexcept { return *base_; }
  MeshBase& base() noexcept { return *base_; }

  // Fast path aliases
  size_t n_nodes() const noexcept { return base_->n_nodes(); }
  size_t n_cells() const noexcept { return base_->n_cells(); }
  const std::vector<real_t>& X_ref() const noexcept { return base_->X_ref(); }
  const std::vector<CellShape>& cell_shapes() const noexcept { return base_->cell_shapes(); }

private:
  std::shared_ptr<MeshBase> base_;
};

} // namespace svmp

#endif // SVMP_MESH_H
