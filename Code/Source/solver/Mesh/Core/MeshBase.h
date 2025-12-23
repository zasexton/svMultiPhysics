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

#ifndef SVMP_MESH_BASE_H
#define SVMP_MESH_BASE_H

#include "MeshTypes.h"
#include "../Topology/CellShape.h"
#include "../Observer/MeshObserver.h"
#include "../Fields/MeshFieldDescriptor.h"
#include "../Search/SearchAccel.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <utility>
#include <string>

namespace svmp {

/**
 * @brief Core mesh container class
 *
 * This class provides the fundamental mesh data structure with:
 * - Topology storage (volumes, faces, edges, vertices)
 * - Coordinate storage (reference and current configurations)
 * - Field attachment system
 * - Label and set management
 * - IO registry
 *
 * Specialized functionality is delegated to component classes:
 * - MeshGeometry: Geometric computations
 * - MeshQuality: Quality metrics
 * - MeshTopology: Adjacency and topology operations
 * - MeshSearch: Point location and search
 * - MeshFields: Field management
 * - MeshLabels: Label and set operations
 */
class MeshBase {
public:
  // ---- Lifecycle ----
  MeshBase();
  explicit MeshBase(int spatial_dim);
  ~MeshBase() = default;

  // Copy and move constructors
  MeshBase(const MeshBase&) = delete;
  MeshBase& operator=(const MeshBase&) = delete;
  MeshBase(MeshBase&&) = default;
  MeshBase& operator=(MeshBase&&) = default;

  // ---- Builders ----
  void clear();
  void reserve(index_t n_vertices, index_t n_cells, index_t n_faces = 0);

  void build_from_arrays(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape);

  void set_faces_from_arrays(
      const std::vector<CellShape>& face_shape,
      const std::vector<offset_t>& face2vertex_offsets,
      const std::vector<index_t>& face2vertex,
      const std::vector<std::array<index_t,2>>& face2cell);
  void set_cell_faces_from_arrays(
      const std::vector<offset_t>& cell2face_offsets,
      const std::vector<index_t>& cell2face,
      const std::vector<int8_t>& cell2face_sense);
  void set_cell_faces_from_arrays(
      const std::vector<offset_t>& cell2face_offsets,
      const std::vector<index_t>& cell2face,
      const std::vector<int8_t>& cell2face_sense,
      const std::vector<int8_t>& cell2face_perm);
  void set_edges_from_arrays(const std::vector<std::array<index_t,2>>& edge2vertex);

  void finalize();

  // ---- Basic queries ----
  int dim() const noexcept { return spatial_dim_; }
  const std::string& mesh_id() const noexcept { return mesh_id_; }
  size_t n_vertices() const noexcept { return X_ref_.size() / (spatial_dim_ > 0 ? spatial_dim_ : 1); }
  size_t n_cells() const noexcept { return cell_shape_.size(); }
  size_t n_faces() const noexcept { return face_shape_.size(); }
  size_t n_edges() const noexcept { return edge2vertex_.size(); }
  // Number of boundary faces (faces with a single incident cell or explicitly marked)
  size_t n_boundary_faces() const { return boundary_faces().size(); }

  // ---- Coordinates ----
  const std::vector<real_t>& X_ref() const noexcept { return X_ref_; }
  const std::vector<real_t>& X_cur() const noexcept { return X_cur_; }
  bool has_current_coords() const noexcept { return !X_cur_.empty(); }
  /// @brief Mutable pointer to current coordinates (advanced use).
  ///
  /// This is intended for internal Mesh-library routines that need to update
  /// a subset of vertex coordinates in-place (e.g., MPI ghost synchronization)
  /// without copying the full coordinate array. Callers are responsible for
  /// emitting a GeometryChanged event after mutating this buffer.
  real_t* X_cur_data_mutable() noexcept { return X_cur_.empty() ? nullptr : X_cur_.data(); }
  void set_current_coords(const std::vector<real_t>& Xcur);
  void clear_current_coords();
  Configuration active_configuration() const noexcept { return active_config_; }
  void use_reference_configuration() { active_config_ = Configuration::Reference; }
  void use_current_configuration() { active_config_ = Configuration::Current; }

  // Convenience vertex coordinate accessors (for small testing workflows)
  std::array<real_t,3> get_vertex_coords(index_t v) const;
  void set_vertex_coords(index_t v, const std::array<real_t,3>& xyz);
  void set_vertex_deformed_coords(index_t v, const std::array<real_t,3>& xyz) { set_vertex_coords(v, xyz); }

  // ---- Topology access ----
  const std::vector<CellShape>& cell_shapes() const noexcept { return cell_shape_; }
  const CellShape& cell_shape(index_t c) const { return cell_shape_.at(static_cast<size_t>(c)); }
  std::pair<const index_t*, size_t> cell_vertices_span(index_t c) const;
  // Convenience: return cell vertices as a vector
  std::vector<index_t> cell_vertices(index_t c) const;
  // Legacy/compat: get full connectivity for a cell
  std::vector<index_t> get_cell_connectivity(index_t c) const { return cell_vertices(c); }
  const std::vector<offset_t>& cell2vertex_offsets() const noexcept { return cell2vertex_offsets_; }
  const std::vector<index_t>& cell2vertex() const noexcept { return cell2vertex_; }

  const std::vector<CellShape>& face_shapes() const noexcept { return face_shape_; }
  std::pair<const index_t*, size_t> face_vertices_span(index_t f) const;
  // Convenience: return face vertices as a vector
  std::vector<index_t> face_vertices(index_t f) const;
  // Faces incident to a cell.
  // - If cell->face adjacency is available, this is O(1) + O(n_faces(cell)).
  // - Otherwise, it falls back to a linear scan of face2cell_.
  std::vector<index_t> cell_faces(index_t c) const;
  std::pair<const index_t*, size_t> cell_faces_span(index_t c) const;
  std::pair<const int8_t*, size_t> cell_face_senses_span(index_t c) const;
  std::pair<const int8_t*, size_t> cell_face_permutations_span(index_t c) const;
  // Zero-copy access to face connectivity in CSR form
  const std::vector<offset_t>& face2vertex_offsets() const noexcept { return face2vertex_offsets_; }
  const std::vector<index_t>& face2vertex() const noexcept { return face2vertex_; }
  // Incident cell IDs for each face (size = n_faces)
  const std::vector<std::array<index_t,2>>& face2cell() const noexcept { return face2cell_; }
  // Boundary labels per face (size = n_faces, INVALID_LABEL = unlabeled)
  const std::vector<label_t>& face_boundary_ids() const noexcept { return face_boundary_id_; }

  // ---- Incremental builders (testing convenience; not optimized for large meshes) ----
  void add_vertex(index_t id, const std::array<real_t,3>& xyz);
  void add_cell(index_t id, CellFamily family, const std::vector<index_t>& vertices);
  void add_boundary_face(index_t id, const std::vector<index_t>& vertices);
  const std::array<index_t,2>& face_cells(index_t f) const { return face2cell_.at(static_cast<size_t>(f)); }

  const std::vector<std::array<index_t,2>>& edge2vertex() const noexcept { return edge2vertex_; }
  const std::array<index_t,2>& edge_vertices(index_t e) const { return edge2vertex_.at(static_cast<size_t>(e)); }

  // ---- IDs / ownership ----
  const std::vector<gid_t>& vertex_gids() const noexcept { return vertex_gid_; }
  const std::vector<gid_t>& cell_gids() const noexcept { return cell_gid_; }
  const std::vector<gid_t>& face_gids() const noexcept { return face_gid_; }
  const std::vector<gid_t>& edge_gids() const noexcept { return edge_gid_; }

  void set_vertex_gids(std::vector<gid_t> gids);
  void set_cell_gids(std::vector<gid_t> gids);
  void set_face_gids(std::vector<gid_t> gids);
  void set_edge_gids(std::vector<gid_t> gids);

  index_t global_to_local_cell(gid_t gid) const;
  index_t global_to_local_vertex(gid_t gid) const;
  index_t global_to_local_face(gid_t gid) const;
  index_t global_to_local_edge(gid_t gid) const;

  // ---- Labels & sets ----
  void set_region_label(index_t cell, label_t label);
  label_t region_label(index_t cell) const;
  std::vector<index_t> cells_with_label(label_t label) const;
  const std::vector<label_t>& cell_region_ids() const noexcept { return cell_region_id_; }

  void set_boundary_label(index_t face, label_t label);
  label_t boundary_label(index_t face) const;
  std::vector<index_t> faces_with_label(label_t label) const;

  void set_edge_label(index_t edge, label_t label);
  label_t edge_label(index_t edge) const;
  std::vector<index_t> edges_with_label(label_t label) const;
  const std::vector<label_t>& edge_label_ids() const noexcept { return edge_label_id_; }

  void set_vertex_label(index_t vertex, label_t label);
  label_t vertex_label(index_t vertex) const;
  std::vector<index_t> vertices_with_label(label_t label) const;
  const std::vector<label_t>& vertex_label_ids() const noexcept { return vertex_label_id_; }

  void add_to_set(EntityKind kind, const std::string& name, index_t id);
  const std::vector<index_t>& get_set(EntityKind kind, const std::string& name) const;
  bool has_set(EntityKind kind, const std::string& name) const;
  void remove_from_set(EntityKind kind, const std::string& name, index_t id);
  void remove_set(EntityKind kind, const std::string& name);
  std::vector<std::string> list_sets(EntityKind kind) const;

  void register_label(const std::string& name, label_t label);
  std::string label_name(label_t label) const;
  label_t label_from_name(const std::string& name) const;
  std::unordered_map<label_t, std::string> list_label_names() const;
  void clear_label_registry();

  // ---- Field attachment ----
  FieldHandle attach_field(EntityKind kind, const std::string& name, FieldScalarType type,
                          size_t components, size_t custom_bytes_per_component = 0);
  FieldHandle attach_field_with_descriptor(EntityKind kind, const std::string& name,
                                          FieldScalarType type, const FieldDescriptor& descriptor);
  bool has_field(EntityKind kind, const std::string& name) const;
  FieldHandle field_handle(EntityKind kind, const std::string& name) const;
  void remove_field(const FieldHandle& h);
  void* field_data(const FieldHandle& h);
  const void* field_data(const FieldHandle& h) const;
  size_t field_components(const FieldHandle& h) const;
  FieldScalarType field_type(const FieldHandle& h) const;
  size_t field_entity_count(const FieldHandle& h) const;
  size_t field_bytes_per_entity(const FieldHandle& h) const;
  const FieldDescriptor* field_descriptor(const FieldHandle& h) const;
  void set_field_descriptor(const FieldHandle& h, const FieldDescriptor& descriptor);
  void resize_fields(EntityKind kind, size_t new_count);

  // Field enumeration
  std::vector<std::string> field_names(EntityKind kind) const;
  void* field_data_by_name(EntityKind kind, const std::string& name);
  const void* field_data_by_name(EntityKind kind, const std::string& name) const;
  size_t field_components_by_name(EntityKind kind, const std::string& name) const;
  FieldScalarType field_type_by_name(EntityKind kind, const std::string& name) const;
  size_t field_bytes_per_component_by_name(EntityKind kind, const std::string& name) const;

  template <typename T>
  T* field_data_as(const FieldHandle& h) { return reinterpret_cast<T*>(field_data(h)); }

  template <typename T>
  const T* field_data_as(const FieldHandle& h) const { return reinterpret_cast<const T*>(field_data(h)); }

  // ---- Geometry operations (delegated) ----
  std::array<real_t,3> cell_center(index_t c, Configuration cfg = Configuration::Reference) const;
  std::array<real_t,3> cell_centroid(index_t c, Configuration cfg = Configuration::Reference) const;
  std::array<real_t,3> face_center(index_t f, Configuration cfg = Configuration::Reference) const;
  std::array<real_t,3> face_normal(index_t f, Configuration cfg = Configuration::Reference) const;
  real_t face_area(index_t f, Configuration cfg = Configuration::Reference) const;
  real_t cell_measure(index_t c, Configuration cfg = Configuration::Reference) const;
  BoundingBox bounding_box(Configuration cfg = Configuration::Reference) const;

  // ---- Quality metrics (delegated) ----
  real_t compute_quality(index_t cell, const std::string& metric = "aspect_ratio") const;
  std::pair<real_t,real_t> global_quality_range(const std::string& metric = "aspect_ratio") const;

  // ---- Adjacency queries ----
  std::vector<index_t> cell_neighbors(index_t c) const;
  std::vector<index_t> vertex_cells(index_t v) const;
  std::vector<index_t> face_neighbors(index_t f) const;
  std::vector<index_t> boundary_faces() const;
  std::vector<index_t> boundary_cells() const;

  void build_vertex2cell();
  void build_vertex2face();
  void build_cell2cell();

  // ---- Submesh extraction ----
  MeshBase extract_submesh_by_region(label_t region_label) const;
  MeshBase extract_submesh_by_regions(const std::vector<label_t>& region_labels) const;
  MeshBase extract_submesh_by_boundary(label_t boundary_label) const;

  // ---- Search & point location ----
  PointLocateResult locate_point(const std::array<real_t,3>& x,
                                Configuration cfg = Configuration::Reference) const;
  std::vector<PointLocateResult> locate_points(const std::vector<std::array<real_t,3>>& X,
                                              Configuration cfg = Configuration::Reference) const;
  RayIntersectResult intersect_ray(const std::array<real_t,3>& origin,
                                  const std::array<real_t,3>& direction,
                                  Configuration cfg = Configuration::Reference) const;
  void build_search_structure(Configuration cfg = Configuration::Reference) const;
  void build_search_structure(const MeshSearch::SearchConfig& config,
                              Configuration cfg = Configuration::Reference) const;
  void clear_search_structure() const;
  bool has_search_structure() const { return search_accel_ && search_accel_->is_valid(); }
  const SearchAccel* search_accel() const noexcept { return search_accel_.get(); }

  // ---- Validation & diagnostics ----
  void validate_basic() const;
  void validate_topology() const;
  void validate_geometry() const;
  void report_statistics() const;
  void write_debug(const std::string& prefix, const std::string& format = "vtu") const;

  // ---- Memory management ----
  void shrink_to_fit();
  size_t memory_usage_bytes() const;

  // ---- Event system ----
  MeshEventBus& event_bus() { return event_bus_; }
  MeshEventBus& event_bus() const { return event_bus_; }

  // ---- IO registry ----
  using LoadFn = std::function<MeshBase(const MeshIOOptions&)>;
  using SaveFn = std::function<void(const MeshBase&, const MeshIOOptions&)>;

  static void register_reader(const std::string& format, LoadFn fn);
  static void register_writer(const std::string& format, SaveFn fn);
  static MeshBase load(const MeshIOOptions& opts);
  void save(const MeshIOOptions& opts) const;
  static std::vector<std::string> registered_readers();
  static std::vector<std::string> registered_writers();

private:
  // Core data
  int spatial_dim_ = 0;
  std::string mesh_id_;
  Configuration active_config_ = Configuration::Reference;

  // Coordinates
  std::vector<real_t> X_ref_;
  std::vector<real_t> X_cur_;

  // Cell topology
  std::vector<CellShape> cell_shape_;
  std::vector<offset_t> cell2vertex_offsets_;
  std::vector<index_t> cell2vertex_;

  // Face topology
  std::vector<CellShape> face_shape_;
  std::vector<offset_t> face2vertex_offsets_;
  std::vector<index_t> face2vertex_;
  std::vector<std::array<index_t,2>> face2cell_;
  // Optional cell->face adjacency (parallel arrays).
  std::vector<offset_t> cell2face_offsets_;
  std::vector<index_t> cell2face_;
  std::vector<int8_t> cell2face_sense_;
  std::vector<int8_t> cell2face_perm_;

  // Edge topology
  std::vector<std::array<index_t,2>> edge2vertex_;

  // Global IDs
  std::vector<gid_t> vertex_gid_;
  std::vector<gid_t> cell_gid_;
  std::vector<gid_t> face_gid_;
  std::vector<gid_t> edge_gid_;

  // Ownership
  std::vector<Ownership> vertex_owner_;
  std::vector<Ownership> cell_owner_;
  std::vector<Ownership> face_owner_;
  std::vector<Ownership> edge_owner_;

  // Labels
  std::vector<label_t> vertex_label_id_;
  std::vector<label_t> cell_region_id_;
  std::vector<label_t> face_boundary_id_;
  std::vector<label_t> edge_label_id_;
  std::unordered_map<std::string, std::vector<index_t>> entity_sets_[4];
  std::unordered_map<std::string, label_t> label_from_name_;
  std::vector<std::string> name_from_label_;

  // Field attachments
  struct FieldInfo {
    uint32_t id = 0;
    FieldScalarType type;
    size_t components;
    size_t bytes_per_component;
    std::vector<uint8_t> data;
  };
  struct AttachTable { std::unordered_map<std::string, FieldInfo> by_name; };
  AttachTable attachments_[4];
  uint32_t next_field_id_ = 1;
  std::unordered_map<uint32_t, std::pair<EntityKind, std::string>> field_index_;
  std::unordered_map<uint32_t, FieldDescriptor> field_descriptors_;

  // Adjacency caches (mutable for lazy building)
  mutable std::vector<offset_t> vertex2cell_offsets_;
  mutable std::vector<index_t> vertex2cell_;
  mutable std::vector<offset_t> vertex2face_offsets_;
  mutable std::vector<index_t> vertex2face_;
  mutable std::vector<offset_t> cell2cell_offsets_;
  mutable std::vector<index_t> cell2cell_;

  // Global to local maps
  std::unordered_map<gid_t, index_t> global2local_cell_;
  std::unordered_map<gid_t, index_t> global2local_vertex_;
  std::unordered_map<gid_t, index_t> global2local_face_;
  std::unordered_map<gid_t, index_t> global2local_edge_;

  // Search acceleration
  mutable std::unique_ptr<SearchAccel> search_accel_;

  // Event bus
  mutable MeshEventBus event_bus_;

  // Helper methods
  size_t entity_count(EntityKind k) const noexcept;
  void invalidate_caches();
  void rebuild_gid_maps();

  // IO registry storage
  static std::unordered_map<std::string, LoadFn>& readers_();
  static std::unordered_map<std::string, SaveFn>& writers_();
};

} // namespace svmp

#endif // SVMP_MESH_BASE_H
