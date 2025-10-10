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

#include <algorithm>
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
using offset_t = int64_t;     // CSR offsets (can handle > 2B entries)
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
      const std::vector<offset_t>& cell2node_offsets,
      const std::vector<index_t>& cell2node,
      const std::vector<CellShape>& cell_shape);
  void set_faces_from_arrays(
      const std::vector<CellShape>& face_shape,
      const std::vector<offset_t>& face2node_offsets,
      const std::vector<index_t>& face2node,
      const std::vector<std::array<index_t,2>>& face2cell);
  void finalize(); // build minimal caches, validate basic invariants

  // ---- Basic queries
  int dim() const noexcept { return spatial_dim_; }
  size_t n_nodes()  const noexcept { return X_ref_.size() / (spatial_dim_ > 0 ? spatial_dim_ : 1); }
  size_t n_cells()  const noexcept { return cell_shape_.size(); }
  size_t n_faces()  const noexcept { return face_shape_.size(); }
  size_t n_edges()  const noexcept { return edge2node_.size(); }

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
  const std::vector<offset_t>& cell2node_offsets() const noexcept { return cell2node_offsets_; }
  const std::vector<index_t>& cell2node() const noexcept { return cell2node_; }

  // Faces (optional if not provided by builder)
  const std::vector<CellShape>& face_shapes() const noexcept { return face_shape_; }
  std::pair<const index_t*, size_t> face_nodes_span(index_t f) const;
  const std::array<index_t,2>& face_cells(index_t f) const { return face2cell_.at(static_cast<size_t>(f)); }

  // Edges (optional topology for high-order FE)
  const std::vector<std::array<index_t,2>>& edge2node() const noexcept { return edge2node_; }
  const std::array<index_t,2>& edge_nodes(index_t e) const { return edge2node_.at(static_cast<size_t>(e)); }
  void set_edges_from_arrays(const std::vector<std::array<index_t,2>>& edge2node);

  // IDs / ownership
  const std::vector<gid_t>& node_gids() const noexcept { return node_gid_; }
  const std::vector<gid_t>& cell_gids() const noexcept { return cell_gid_; }
  const std::vector<gid_t>& face_gids() const noexcept { return face_gid_; }
  const std::vector<gid_t>& edge_gids() const noexcept { return edge_gid_; }
  void set_node_gids(std::vector<gid_t> gids) { node_gid_ = std::move(gids); }
  void set_cell_gids(std::vector<gid_t> gids) { cell_gid_ = std::move(gids); }
  void set_face_gids(std::vector<gid_t> gids) { face_gid_ = std::move(gids); }
  void set_edge_gids(std::vector<gid_t> gids) { edge_gid_ = std::move(gids); }

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
  FieldHandle attach_field(EntityKind kind, const std::string& name, FieldScalarType type,
                           size_t components, size_t custom_bytes_per_component = 0);
  bool has_field(EntityKind kind, const std::string& name) const;
  void remove_field(const FieldHandle& h);
  void* field_data(const FieldHandle& h);
  const void* field_data(const FieldHandle& h) const;
  size_t field_components(const FieldHandle& h) const;
  FieldScalarType field_type(const FieldHandle& h) const;
  size_t field_entity_count(const FieldHandle& h) const;
  size_t field_bytes_per_entity(const FieldHandle& h) const;

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
  void validate_topology() const; // duplicate nodes, degenerate cells, inverted elements
  void validate_geometry() const; // face orientation consistency, normals
  void report_statistics() const; // mesh quality summary
  void write_debug_vtk(const std::string& prefix) const;

  // ---- Global ID management (parallel-friendly but not MPI-dependent)
  index_t global_to_local_cell(gid_t gid) const;
  index_t global_to_local_node(gid_t gid) const;
  index_t global_to_local_face(gid_t gid) const;

  // ---- Adjacency queries
  std::vector<index_t> cell_neighbors(index_t c) const;
  std::vector<index_t> node_cells(index_t n) const;
  std::vector<index_t> face_neighbors(index_t f) const;
  std::vector<index_t> boundary_faces() const;
  std::vector<index_t> boundary_cells() const;

  // Build adjacency caches
  void build_node2cell();
  void build_node2face();
  void build_cell2cell();

  // ---- Submesh extraction
  MeshBase extract_submesh_by_region(label_t region_label) const;
  MeshBase extract_submesh_by_regions(const std::vector<label_t>& region_labels) const;
  MeshBase extract_submesh_by_boundary(label_t boundary_label) const;

  template <typename Predicate>
  MeshBase extract_submesh_by_predicate(Predicate pred) const {
    // Extract cells matching predicate
    std::vector<index_t> cells_to_extract;
    for (size_t c = 0; c < n_cells(); ++c) {
      if (pred(static_cast<index_t>(c))) {
        cells_to_extract.push_back(static_cast<index_t>(c));
      }
    }

    // Extract unique nodes
    std::unordered_set<index_t> node_set;
    for (index_t c : cells_to_extract) {
      auto [nodes_ptr, n_node] = cell_nodes_span(c);
      for (size_t i = 0; i < n_node; ++i) {
        node_set.insert(nodes_ptr[i]);
      }
    }

    std::vector<index_t> nodes_to_extract(node_set.begin(), node_set.end());
    std::sort(nodes_to_extract.begin(), nodes_to_extract.end());

    // Create node mapping
    std::unordered_map<index_t, index_t> old_to_new_node;
    for (size_t i = 0; i < nodes_to_extract.size(); ++i) {
      old_to_new_node[nodes_to_extract[i]] = static_cast<index_t>(i);
    }

    // Build submesh
    MeshBase submesh(spatial_dim_);

    // Copy node coordinates
    std::vector<real_t> new_coords(nodes_to_extract.size() * spatial_dim_);
    for (size_t i = 0; i < nodes_to_extract.size(); ++i) {
      index_t old_node = nodes_to_extract[i];
      for (int d = 0; d < spatial_dim_; ++d) {
        new_coords[i * spatial_dim_ + d] = X_ref_[old_node * spatial_dim_ + d];
      }
    }

    // Build cell connectivity
    std::vector<CellShape> new_cell_shapes;
    std::vector<offset_t> new_cell2node_offsets = {0};
    std::vector<index_t> new_cell2node;

    for (index_t c : cells_to_extract) {
      new_cell_shapes.push_back(cell_shape_[c]);

      auto [nodes_ptr, n_node] = cell_nodes_span(c);
      for (size_t i = 0; i < n_node; ++i) {
        new_cell2node.push_back(old_to_new_node[nodes_ptr[i]]);
      }
      new_cell2node_offsets.push_back(static_cast<offset_t>(new_cell2node.size()));
    }

    submesh.build_from_arrays(spatial_dim_, new_coords, new_cell2node_offsets,
                             new_cell2node, new_cell_shapes);

    return submesh;
  }

  // ---- Advanced geometry & mapping
  std::array<real_t,3> evaluate_map(index_t cell, const std::array<real_t,3>& xi, Configuration cfg = Configuration::Reference) const;
  std::array<std::array<real_t,3>,3> jacobian(index_t cell, const std::array<real_t,3>& xi, Configuration cfg = Configuration::Reference) const;
  real_t detJ(index_t cell, const std::array<real_t,3>& xi, Configuration cfg = Configuration::Reference) const;
  std::array<std::array<real_t,3>,3> invJ(index_t cell, const std::array<real_t,3>& xi, Configuration cfg = Configuration::Reference) const;

  std::array<real_t,3> face_normal(index_t f, Configuration cfg = Configuration::Reference) const;
  std::array<real_t,3> face_center(index_t f, Configuration cfg = Configuration::Reference) const;
  real_t face_area(index_t f, Configuration cfg = Configuration::Reference) const;
  real_t cell_measure(index_t c, Configuration cfg = Configuration::Reference) const;

  void use_reference_configuration();
  void use_current_configuration();
  void set_displacement_field(const FieldHandle& U);
  Configuration active_configuration() const noexcept { return active_config_; }

  // ---- Quality metrics
  real_t compute_quality(index_t cell, const std::string& metric = "aspect_ratio") const;
  std::pair<real_t,real_t> global_quality_range(const std::string& metric = "aspect_ratio") const;

  // ---- Search structures & point location
  struct PointLocateResult {
    index_t cell_id = -1;
    std::array<real_t,3> xi = {{0,0,0}};
    bool found = false;
  };

  PointLocateResult locate_point(const std::array<real_t,3>& x, Configuration cfg = Configuration::Reference) const;
  std::vector<PointLocateResult> locate_points(const std::vector<std::array<real_t,3>>& X, Configuration cfg = Configuration::Reference) const;

  struct RayIntersectResult {
    index_t face_id = -1;
    real_t t = -1.0;
    std::array<real_t,3> point = {{0,0,0}};
    bool found = false;
  };

  RayIntersectResult intersect_ray(const std::array<real_t,3>& origin, const std::array<real_t,3>& direction, Configuration cfg = Configuration::Reference) const;

  void build_search_structure(Configuration cfg = Configuration::Reference) const;
  void clear_search_structure() const;

  // ---- Adaptivity methods
  void mark_refine(const std::vector<index_t>& cells);
  void mark_coarsen(const std::vector<index_t>& cells);
  void apply_refinement();
  void apply_coarsening();
  index_t parent(index_t cell) const;
  std::vector<index_t> children(index_t cell) const;

  // ---- Periodic/mortar/contact constraints
  void register_periodic_pair(const std::string& face_set_A, const std::string& face_set_B, const std::array<real_t,9>& transform);
  void register_mortar_interface(const std::string& master_set, const std::string& slave_set);
  void register_contact_candidates(const std::string& setA, const std::string& setB);

  // ---- Label <-> name bidirectional registry
  void register_label(const std::string& name, label_t label);
  std::string label_name(label_t label) const;
  label_t label_from_name(const std::string& name) const;

  // ---- Memory management
  void shrink_to_fit();
  size_t memory_usage_bytes() const;

  // ---- Builder utilities
  static MeshBase build_cartesian(int nx, int ny, int nz, const BoundingBox& domain);
  static MeshBase build_extruded(const MeshBase& base_2d, int n_layers, real_t height);

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
  real_t tet_volume(const std::array<real_t,3>& p0, const std::array<real_t,3>& p1,
                    const std::array<real_t,3>& p2, const std::array<real_t,3>& p3) const;

  // Search acceleration structure (forward declaration, impl in cpp)
  struct SearchAccel;
  mutable std::unique_ptr<SearchAccel> search_accel_;

private:
  int spatial_dim_ = 0;
  Configuration active_config_ = Configuration::Reference;

  // Geometry
  std::vector<real_t> X_ref_;  // size = n_nodes * spatial_dim
  std::vector<real_t> X_cur_;  // optional current (ALE/FSI) same size as X_ref

  // Topology: cells
  std::vector<CellShape>  cell_shape_;
  std::vector<offset_t>   cell2node_offsets_; // CSR offsets size n_cells+1
  std::vector<index_t>    cell2node_;         // concatenated node indices

  // Topology: faces (optional)
  std::vector<CellShape>  face_shape_;
  std::vector<offset_t>   face2node_offsets_;
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
  struct FieldInfo {
    FieldScalarType type;
    size_t components;
    size_t bytes_per_component; // for Custom types
    std::vector<uint8_t> data;
  };
  struct AttachTable { std::unordered_map<std::string, FieldInfo> by_name; };
  AttachTable attachments_[4];
  uint32_t next_field_id_ = 1;
  std::unordered_map<uint32_t, std::pair<EntityKind, std::string>> field_index_; // id -> (kind,name)

  // Adaptivity (placeholders)
  std::vector<index_t> cell_parent_;                  // parent cell id or -1 if root
  std::vector<offset_t> cell_children_offsets_;       // CSR offsets into cell_children_
  std::vector<index_t> cell_children_;                // concatenated children ids
  std::vector<uint8_t> refine_flag_;                  // 1=refine, 0=keep
  std::vector<uint8_t> coarsen_flag_;                 // 1=coarsen, 0=keep

  // Local<->Global maps
  std::unordered_map<gid_t, index_t> global2local_cell_;
  std::unordered_map<gid_t, index_t> global2local_node_;
  std::unordered_map<gid_t, index_t> global2local_face_;

  // Adjacency caches (lazily built)
  mutable std::vector<offset_t> node2cell_offsets_;   // CSR offsets
  mutable std::vector<index_t> node2cell_;            // concatenated cell ids
  mutable std::vector<offset_t> node2face_offsets_;   // CSR offsets
  mutable std::vector<index_t> node2face_;            // concatenated face ids
  mutable std::vector<offset_t> cell2cell_offsets_;   // CSR offsets
  mutable std::vector<index_t> cell2cell_;            // concatenated cell ids

  // Edge topology (optional, for high-order FE)
  std::vector<std::array<index_t,2>> edge2node_;     // 2 nodes per edge
  std::vector<gid_t> edge_gid_;
  std::vector<Ownership> edge_owner_;

  // High-order/curvilinear geometry support
  std::vector<int> cell_geom_order_;                 // polynomial order per cell
  std::vector<offset_t> cell2geomnode_offsets_;      // CSR for geometry nodes
  std::vector<index_t> cell2geomnode_;               // node IDs for geometry mapping

  // Label <-> name bidirectional registry
  std::unordered_map<std::string, label_t> label_from_name_;
  std::vector<std::string> name_from_label_;

  // Cached geometry (invalidated on modification)
  mutable std::vector<real_t> cell_measures_;        // length/area/volume per cell
  mutable std::vector<real_t> face_areas_;           // area per face
  mutable std::vector<std::array<real_t,3>> face_normals_; // normal per face
  mutable bool geom_cache_valid_ = false;

  // Periodic/mortar/contact constraints
  struct PeriodicPair {
    std::string face_set_A;
    std::string face_set_B;
    std::array<real_t,9> transform;
  };
  std::vector<PeriodicPair> periodic_pairs_;

  struct MortarInterface {
    std::string master_set;
    std::string slave_set;
  };
  std::vector<MortarInterface> mortar_interfaces_;

  struct ContactPair {
    std::string setA;
    std::string setB;
  };
  std::vector<ContactPair> contact_candidates_;

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
