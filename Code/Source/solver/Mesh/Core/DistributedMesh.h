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

#ifndef SVMP_DISTRIBUTED_MESH_H
#define SVMP_DISTRIBUTED_MESH_H

#include "MeshBase.h"
#include "MeshComm.h"
#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif
#include <memory>
#include <unordered_set>
#include <vector>

namespace svmp {

#if !defined(MESH_HAS_MPI)

// ------------------------
// Serial stub for DistributedMesh (composition-based, matches MPI version API)
// ------------------------
class DistributedMesh {
public:
  // Constructors - always create internal MeshBase for composition pattern
  DistributedMesh() : local_mesh_(std::make_shared<MeshBase>()) {}

  explicit DistributedMesh(MeshComm) : local_mesh_(std::make_shared<MeshBase>()) {}

  explicit DistributedMesh(std::shared_ptr<MeshBase> local_mesh, MeshComm = MeshComm::world())
    : local_mesh_(local_mesh ? std::move(local_mesh) : std::make_shared<MeshBase>()) {}

  // Access underlying mesh - always dereference, never fallback to *this
  MeshBase& local_mesh() { return *local_mesh_; }
  const MeshBase& local_mesh() const { return *local_mesh_; }
  std::shared_ptr<MeshBase> local_mesh_ptr() { return local_mesh_; }
  std::shared_ptr<const MeshBase> local_mesh_ptr() const { return local_mesh_; }

  // ---- MeshBase-like surface (local meaning)
  //
  // These forwarders allow user code to treat DistributedMesh like MeshBase for
  // common operations, without explicitly calling local_mesh().
  using LoadFn = MeshBase::LoadFn;
  using SaveFn = MeshBase::SaveFn;

  MeshBase& base() { return local_mesh(); }
  const MeshBase& base() const { return local_mesh(); }

  // Lifecycle/builders
  void clear() {
    local_mesh().clear();
    reset_partition_state_();
  }
  void reserve(index_t n_vertices, index_t n_cells, index_t n_faces = 0) {
    local_mesh().reserve(n_vertices, n_cells, n_faces);
  }
  // Builders
  // - `build_from_arrays(...)` is a *local* build in all builds (serial + MPI).
  // - Use `build_from_arrays_global_and_partition(...)` to build from a global mesh
  //   on rank 0 and distribute/partition it across ranks.
  void build_from_arrays_local(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape) {
    local_mesh().build_from_arrays(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
    reset_partition_state_();
  }
  void build_from_arrays(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape) {
    build_from_arrays_local(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
  }

  void build_from_arrays_global_and_partition(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape,
      PartitionHint = PartitionHint::Cells,
      int ghost_layers = 0,
      const std::unordered_map<std::string, std::string>& = {}) {
    (void)ghost_layers;
    build_from_arrays_local(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
  }
  void set_faces_from_arrays(
      const std::vector<CellShape>& face_shape,
      const std::vector<offset_t>& face2vertex_offsets,
      const std::vector<index_t>& face2vertex,
      const std::vector<std::array<index_t,2>>& face2cell) {
    local_mesh().set_faces_from_arrays(face_shape, face2vertex_offsets, face2vertex, face2cell);
    reset_partition_state_();
  }
  void set_cell_faces_from_arrays(
      const std::vector<offset_t>& cell2face_offsets,
      const std::vector<index_t>& cell2face,
      const std::vector<int8_t>& cell2face_sense) {
    local_mesh().set_cell_faces_from_arrays(cell2face_offsets, cell2face, cell2face_sense);
    reset_partition_state_();
  }
  void set_cell_faces_from_arrays(
      const std::vector<offset_t>& cell2face_offsets,
      const std::vector<index_t>& cell2face,
      const std::vector<int8_t>& cell2face_sense,
      const std::vector<int8_t>& cell2face_perm) {
    local_mesh().set_cell_faces_from_arrays(cell2face_offsets, cell2face, cell2face_sense, cell2face_perm);
    reset_partition_state_();
  }
  void set_edges_from_arrays(const std::vector<std::array<index_t,2>>& edge2vertex) {
    local_mesh().set_edges_from_arrays(edge2vertex);
    reset_partition_state_();
  }
  void finalize() {
    local_mesh().finalize();
    reset_partition_state_();
  }

  // Basic queries
  int dim() const noexcept { return local_mesh().dim(); }
  const std::string& mesh_id() const noexcept { return local_mesh().mesh_id(); }
  size_t n_vertices() const noexcept { return local_mesh().n_vertices(); }
  size_t n_cells() const noexcept { return local_mesh().n_cells(); }
  size_t n_faces() const noexcept { return local_mesh().n_faces(); }
  size_t n_edges() const noexcept { return local_mesh().n_edges(); }
  size_t n_boundary_faces() const { return local_mesh().n_boundary_faces(); }

  // Coordinates
  const std::vector<real_t>& X_ref() const noexcept { return local_mesh().X_ref(); }
  const std::vector<real_t>& X_cur() const noexcept { return local_mesh().X_cur(); }
  bool has_current_coords() const noexcept { return local_mesh().has_current_coords(); }
  real_t* X_cur_data_mutable() noexcept { return local_mesh().X_cur_data_mutable(); }
  void set_current_coords(const std::vector<real_t>& Xcur) { local_mesh().set_current_coords(Xcur); }
  void clear_current_coords() { local_mesh().clear_current_coords(); }
  Configuration active_configuration() const noexcept { return local_mesh().active_configuration(); }
  void use_reference_configuration() { local_mesh().use_reference_configuration(); }
  void use_current_configuration() { local_mesh().use_current_configuration(); }
  std::array<real_t,3> get_vertex_coords(index_t v) const { return local_mesh().get_vertex_coords(v); }
  void set_vertex_coords(index_t v, const std::array<real_t,3>& xyz) { local_mesh().set_vertex_coords(v, xyz); }
  void set_vertex_deformed_coords(index_t v, const std::array<real_t,3>& xyz) {
    local_mesh().set_vertex_deformed_coords(v, xyz);
  }

  // Topology access
  const std::vector<CellShape>& cell_shapes() const noexcept { return local_mesh().cell_shapes(); }
  const CellShape& cell_shape(index_t c) const { return local_mesh().cell_shape(c); }
  std::pair<const index_t*, size_t> cell_vertices_span(index_t c) const { return local_mesh().cell_vertices_span(c); }
  std::vector<index_t> cell_vertices(index_t c) const { return local_mesh().cell_vertices(c); }
  std::vector<index_t> get_cell_connectivity(index_t c) const { return local_mesh().get_cell_connectivity(c); }
  const std::vector<offset_t>& cell2vertex_offsets() const noexcept { return local_mesh().cell2vertex_offsets(); }
  const std::vector<index_t>& cell2vertex() const noexcept { return local_mesh().cell2vertex(); }

  const std::vector<CellShape>& face_shapes() const noexcept { return local_mesh().face_shapes(); }
  std::pair<const index_t*, size_t> face_vertices_span(index_t f) const { return local_mesh().face_vertices_span(f); }
  std::vector<index_t> face_vertices(index_t f) const { return local_mesh().face_vertices(f); }
  std::vector<index_t> cell_faces(index_t c) const { return local_mesh().cell_faces(c); }
  std::pair<const index_t*, size_t> cell_faces_span(index_t c) const { return local_mesh().cell_faces_span(c); }
  std::pair<const int8_t*, size_t> cell_face_senses_span(index_t c) const {
    return local_mesh().cell_face_senses_span(c);
  }
  std::pair<const int8_t*, size_t> cell_face_permutations_span(index_t c) const {
    return local_mesh().cell_face_permutations_span(c);
  }
  const std::vector<offset_t>& face2vertex_offsets() const noexcept { return local_mesh().face2vertex_offsets(); }
  const std::vector<index_t>& face2vertex() const noexcept { return local_mesh().face2vertex(); }
  const std::vector<std::array<index_t,2>>& face2cell() const noexcept { return local_mesh().face2cell(); }
  const std::vector<label_t>& face_boundary_ids() const noexcept { return local_mesh().face_boundary_ids(); }

  // Incremental builders
  void add_vertex(index_t id, const std::array<real_t,3>& xyz) {
    local_mesh().add_vertex(id, xyz);
    reset_partition_state_();
  }
  void add_cell(index_t id, CellFamily family, const std::vector<index_t>& vertices) {
    local_mesh().add_cell(id, family, vertices);
    reset_partition_state_();
  }
  void add_boundary_face(index_t id, const std::vector<index_t>& vertices) {
    local_mesh().add_boundary_face(id, vertices);
    reset_partition_state_();
  }
  const std::array<index_t,2>& face_cells(index_t f) const { return local_mesh().face_cells(f); }
  const std::vector<std::array<index_t,2>>& edge2vertex() const noexcept { return local_mesh().edge2vertex(); }
  const std::array<index_t,2>& edge_vertices(index_t e) const { return local_mesh().edge_vertices(e); }

  // IDs / ownership
  const std::vector<gid_t>& vertex_gids() const noexcept { return local_mesh().vertex_gids(); }
  const std::vector<gid_t>& cell_gids() const noexcept { return local_mesh().cell_gids(); }
  const std::vector<gid_t>& face_gids() const noexcept { return local_mesh().face_gids(); }
  const std::vector<gid_t>& edge_gids() const noexcept { return local_mesh().edge_gids(); }
  void set_vertex_gids(std::vector<gid_t> gids) {
    local_mesh().set_vertex_gids(std::move(gids));
    invalidate_exchange_patterns_();
  }
  void set_cell_gids(std::vector<gid_t> gids) {
    local_mesh().set_cell_gids(std::move(gids));
    invalidate_exchange_patterns_();
  }
  void set_face_gids(std::vector<gid_t> gids) {
    local_mesh().set_face_gids(std::move(gids));
    invalidate_exchange_patterns_();
  }
  void set_edge_gids(std::vector<gid_t> gids) { local_mesh().set_edge_gids(std::move(gids)); }
  index_t global_to_local_cell(gid_t gid) const { return local_mesh().global_to_local_cell(gid); }
  index_t global_to_local_vertex(gid_t gid) const { return local_mesh().global_to_local_vertex(gid); }
  index_t global_to_local_face(gid_t gid) const { return local_mesh().global_to_local_face(gid); }
  index_t global_to_local_edge(gid_t gid) const { return local_mesh().global_to_local_edge(gid); }

  // Adaptivity metadata
  size_t refinement_level(index_t cell) const { return local_mesh().refinement_level(cell); }
  void set_refinement_level(index_t cell, size_t level) { local_mesh().set_refinement_level(cell, level); }
  const std::vector<size_t>& cell_refinement_levels() const noexcept { return local_mesh().cell_refinement_levels(); }
  void set_cell_refinement_levels(std::vector<size_t> levels) {
    local_mesh().set_cell_refinement_levels(std::move(levels));
  }

  // Labels & sets
  void set_region_label(index_t cell, label_t label) { local_mesh().set_region_label(cell, label); }
  label_t region_label(index_t cell) const { return local_mesh().region_label(cell); }
  std::vector<index_t> cells_with_label(label_t label) const { return local_mesh().cells_with_label(label); }
  const std::vector<label_t>& cell_region_ids() const noexcept { return local_mesh().cell_region_ids(); }

  void set_boundary_label(index_t face, label_t label) { local_mesh().set_boundary_label(face, label); }
  label_t boundary_label(index_t face) const { return local_mesh().boundary_label(face); }
  std::vector<index_t> faces_with_label(label_t label) const { return local_mesh().faces_with_label(label); }

  void set_edge_label(index_t edge, label_t label) { local_mesh().set_edge_label(edge, label); }
  label_t edge_label(index_t edge) const { return local_mesh().edge_label(edge); }
  std::vector<index_t> edges_with_label(label_t label) const { return local_mesh().edges_with_label(label); }
  const std::vector<label_t>& edge_label_ids() const noexcept { return local_mesh().edge_label_ids(); }

  void set_vertex_label(index_t vertex, label_t label) { local_mesh().set_vertex_label(vertex, label); }
  label_t vertex_label(index_t vertex) const { return local_mesh().vertex_label(vertex); }
  std::vector<index_t> vertices_with_label(label_t label) const { return local_mesh().vertices_with_label(label); }
  const std::vector<label_t>& vertex_label_ids() const noexcept { return local_mesh().vertex_label_ids(); }

  void add_to_set(EntityKind kind, const std::string& name, index_t id) { local_mesh().add_to_set(kind, name, id); }
  const std::vector<index_t>& get_set(EntityKind kind, const std::string& name) const {
    return local_mesh().get_set(kind, name);
  }
  bool has_set(EntityKind kind, const std::string& name) const { return local_mesh().has_set(kind, name); }
  void remove_from_set(EntityKind kind, const std::string& name, index_t id) {
    local_mesh().remove_from_set(kind, name, id);
  }
  void remove_set(EntityKind kind, const std::string& name) { local_mesh().remove_set(kind, name); }
  std::vector<std::string> list_sets(EntityKind kind) const { return local_mesh().list_sets(kind); }

  void register_label(const std::string& name, label_t label) { local_mesh().register_label(name, label); }
  std::string label_name(label_t label) const { return local_mesh().label_name(label); }
  label_t label_from_name(const std::string& name) const { return local_mesh().label_from_name(name); }
  std::unordered_map<label_t, std::string> list_label_names() const { return local_mesh().list_label_names(); }
  void clear_label_registry() { local_mesh().clear_label_registry(); }

  // Fields
  FieldHandle attach_field(EntityKind kind, const std::string& name, FieldScalarType type,
                           size_t components, size_t custom_bytes_per_component = 0) {
    return local_mesh().attach_field(kind, name, type, components, custom_bytes_per_component);
  }
  FieldHandle attach_field_with_descriptor(EntityKind kind, const std::string& name,
                                           FieldScalarType type, const FieldDescriptor& descriptor) {
    return local_mesh().attach_field_with_descriptor(kind, name, type, descriptor);
  }
  bool has_field(EntityKind kind, const std::string& name) const { return local_mesh().has_field(kind, name); }
  FieldHandle field_handle(EntityKind kind, const std::string& name) const { return local_mesh().field_handle(kind, name); }
  void remove_field(const FieldHandle& h) { local_mesh().remove_field(h); }
  void* field_data(const FieldHandle& h) { return local_mesh().field_data(h); }
  const void* field_data(const FieldHandle& h) const { return local_mesh().field_data(h); }
  size_t field_components(const FieldHandle& h) const { return local_mesh().field_components(h); }
  FieldScalarType field_type(const FieldHandle& h) const { return local_mesh().field_type(h); }
  size_t field_entity_count(const FieldHandle& h) const { return local_mesh().field_entity_count(h); }
  size_t field_bytes_per_entity(const FieldHandle& h) const { return local_mesh().field_bytes_per_entity(h); }
  const FieldDescriptor* field_descriptor(const FieldHandle& h) const { return local_mesh().field_descriptor(h); }
  void set_field_descriptor(const FieldHandle& h, const FieldDescriptor& descriptor) {
    local_mesh().set_field_descriptor(h, descriptor);
  }
  void resize_fields(EntityKind kind, size_t new_count) { local_mesh().resize_fields(kind, new_count); }
  std::vector<std::string> field_names(EntityKind kind) const { return local_mesh().field_names(kind); }
  void* field_data_by_name(EntityKind kind, const std::string& name) { return local_mesh().field_data_by_name(kind, name); }
  const void* field_data_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_data_by_name(kind, name);
  }
  size_t field_components_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_components_by_name(kind, name);
  }
  FieldScalarType field_type_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_type_by_name(kind, name);
  }
  size_t field_bytes_per_component_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_bytes_per_component_by_name(kind, name);
  }
  template <typename T>
  T* field_data_as(const FieldHandle& h) { return local_mesh().field_data_as<T>(h); }
  template <typename T>
  const T* field_data_as(const FieldHandle& h) const { return local_mesh().field_data_as<T>(h); }

  // Geometry operations
  std::array<real_t,3> cell_center(index_t c, Configuration cfg = Configuration::Reference) const {
    return local_mesh().cell_center(c, cfg);
  }
  std::array<real_t,3> cell_centroid(index_t c, Configuration cfg = Configuration::Reference) const {
    return local_mesh().cell_centroid(c, cfg);
  }
  std::array<real_t,3> face_center(index_t f, Configuration cfg = Configuration::Reference) const {
    return local_mesh().face_center(f, cfg);
  }
  std::array<real_t,3> face_normal(index_t f, Configuration cfg = Configuration::Reference) const {
    return local_mesh().face_normal(f, cfg);
  }
  real_t face_area(index_t f, Configuration cfg = Configuration::Reference) const { return local_mesh().face_area(f, cfg); }
  real_t cell_measure(index_t c, Configuration cfg = Configuration::Reference) const { return local_mesh().cell_measure(c, cfg); }
  BoundingBox bounding_box(Configuration cfg = Configuration::Reference) const { return local_mesh().bounding_box(cfg); }

  // Quality metrics
  real_t compute_quality(index_t cell, const std::string& metric = "aspect_ratio") const {
    return local_mesh().compute_quality(cell, metric);
  }
  std::pair<real_t,real_t> global_quality_range(const std::string& metric = "aspect_ratio") const {
    return local_mesh().global_quality_range(metric);
  }

  // Adjacency queries
  std::vector<index_t> cell_neighbors(index_t c) const { return local_mesh().cell_neighbors(c); }
  std::vector<index_t> vertex_cells(index_t v) const { return local_mesh().vertex_cells(v); }
  std::vector<index_t> face_neighbors(index_t f) const { return local_mesh().face_neighbors(f); }
  std::vector<index_t> boundary_faces() const { return local_mesh().boundary_faces(); }
  std::vector<index_t> boundary_cells() const { return local_mesh().boundary_cells(); }
  void build_vertex2cell() { local_mesh().build_vertex2cell(); }
  void build_vertex2face() { local_mesh().build_vertex2face(); }
  void build_cell2cell() { local_mesh().build_cell2cell(); }

  // Submesh extraction
  MeshBase extract_submesh_by_region(label_t region_label) const { return local_mesh().extract_submesh_by_region(region_label); }
  MeshBase extract_submesh_by_regions(const std::vector<label_t>& region_labels) const {
    return local_mesh().extract_submesh_by_regions(region_labels);
  }
  MeshBase extract_submesh_by_boundary(label_t boundary_label) const { return local_mesh().extract_submesh_by_boundary(boundary_label); }

  // Search & point location
  PointLocateResult locate_point(const std::array<real_t,3>& x, Configuration cfg = Configuration::Reference) const {
    return local_mesh().locate_point(x, cfg);
  }
  std::vector<PointLocateResult> locate_points(const std::vector<std::array<real_t,3>>& X,
                                               Configuration cfg = Configuration::Reference) const {
    return local_mesh().locate_points(X, cfg);
  }
  RayIntersectResult intersect_ray(const std::array<real_t,3>& origin,
                                   const std::array<real_t,3>& direction,
                                   Configuration cfg = Configuration::Reference) const {
    return local_mesh().intersect_ray(origin, direction, cfg);
  }
  void build_search_structure(Configuration cfg = Configuration::Reference) const { local_mesh().build_search_structure(cfg); }
  void build_search_structure(const MeshSearch::SearchConfig& config,
                              Configuration cfg = Configuration::Reference) const {
    local_mesh().build_search_structure(config, cfg);
  }
  void clear_search_structure() const { local_mesh().clear_search_structure(); }
  bool has_search_structure() const { return local_mesh().has_search_structure(); }
  const SearchAccel* search_accel() const noexcept { return local_mesh().search_accel(); }

  // Validation & diagnostics
  void validate_basic() const { local_mesh().validate_basic(); }
  void validate_topology() const { local_mesh().validate_topology(); }
  void validate_geometry() const { local_mesh().validate_geometry(); }
  void report_statistics() const { local_mesh().report_statistics(); }
  void write_debug(const std::string& prefix, const std::string& format = "vtu") const {
    local_mesh().write_debug(prefix, format);
  }

  // Memory management
  void shrink_to_fit() { local_mesh().shrink_to_fit(); }
  size_t memory_usage_bytes() const { return local_mesh().memory_usage_bytes(); }

  // Event system
  MeshEventBus& event_bus() { return local_mesh().event_bus(); }
  MeshEventBus& event_bus() const { return local_mesh().event_bus(); }

  // I/O (local meaning)
  void save(const MeshIOOptions& opts) const { local_mesh().save(opts); }

  // MPI info
  rank_t rank() const noexcept { return 0; }
  int world_size() const noexcept { return 1; }
  const std::unordered_set<rank_t>& neighbor_ranks() const noexcept { return neighbor_ranks_; }
  // Provide a stub mpi_comm() that can be compared to MPI constants
  // When MPI is not present, return a void* null; pointer comparisons with MPI_Comm work via implicit conversion in tests.
  void* mpi_comm() const noexcept { return nullptr; }
  void set_mpi_comm(MeshComm) {}

  // Ownership (default Owned for all)
  bool is_owned_cell(index_t i) const { return get_owner(cell_owner_, Ownership::Owned, i) == Ownership::Owned; }
  bool is_ghost_cell(index_t i) const { return get_owner(cell_owner_, Ownership::Owned, i) == Ownership::Ghost; }
  bool is_shared_cell(index_t i) const { return get_owner(cell_owner_, Ownership::Owned, i) == Ownership::Shared; }
  rank_t owner_rank_cell(index_t i) const { return get_owner_rank(cell_owner_rank_, i); }

  bool is_owned_vertex(index_t i) const { return get_owner(vertex_owner_, Ownership::Owned, i) == Ownership::Owned; }
  bool is_ghost_vertex(index_t i) const { return get_owner(vertex_owner_, Ownership::Owned, i) == Ownership::Ghost; }
  bool is_shared_vertex(index_t i) const { return get_owner(vertex_owner_, Ownership::Owned, i) == Ownership::Shared; }
  rank_t owner_rank_vertex(index_t i) const { return get_owner_rank(vertex_owner_rank_, i); }

	  bool is_owned_face(index_t i) const { return get_owner(face_owner_, Ownership::Owned, i) == Ownership::Owned; }
	  bool is_ghost_face(index_t i) const { return get_owner(face_owner_, Ownership::Owned, i) == Ownership::Ghost; }
	  bool is_shared_face(index_t i) const { return get_owner(face_owner_, Ownership::Owned, i) == Ownership::Shared; }
	  rank_t owner_rank_face(index_t i) const { return get_owner_rank(face_owner_rank_, i); }

  // ---- Explicit distributed semantics (Phase 3)
  // Default `n_*()` methods on DistributedMesh refer to the *local* mesh size,
  // which includes ghosts when a ghost layer is present. Use these helpers for
  // unambiguous owned/shared/ghost counts and iteration.

  size_t n_owned_vertices() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_vertices());
    for (index_t v = 0; v < n; ++v) {
      if (is_owned_vertex(v)) ++count;
    }
    return count;
  }
  size_t n_shared_vertices() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_vertices());
    for (index_t v = 0; v < n; ++v) {
      if (is_shared_vertex(v)) ++count;
    }
    return count;
  }
  size_t n_ghost_vertices() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_vertices());
    for (index_t v = 0; v < n; ++v) {
      if (is_ghost_vertex(v)) ++count;
    }
    return count;
  }

  size_t n_owned_cells() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_cells());
    for (index_t c = 0; c < n; ++c) {
      if (is_owned_cell(c)) ++count;
    }
    return count;
  }
  size_t n_shared_cells() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_cells());
    for (index_t c = 0; c < n; ++c) {
      if (is_shared_cell(c)) ++count;
    }
    return count;
  }
  size_t n_ghost_cells() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_cells());
    for (index_t c = 0; c < n; ++c) {
      if (is_ghost_cell(c)) ++count;
    }
    return count;
  }

  size_t n_owned_faces() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_faces());
    for (index_t f = 0; f < n; ++f) {
      if (is_owned_face(f)) ++count;
    }
    return count;
  }
  size_t n_shared_faces() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_faces());
    for (index_t f = 0; f < n; ++f) {
      if (is_shared_face(f)) ++count;
    }
    return count;
  }
  size_t n_ghost_faces() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_faces());
    for (index_t f = 0; f < n; ++f) {
      if (is_ghost_face(f)) ++count;
    }
    return count;
  }

  size_t n_ghost_edges() const noexcept {
    size_t count = 0;
    for (const auto& o : edge_owner_) {
      if (o == Ownership::Ghost) ++count;
    }
    return count;
  }
  size_t n_shared_edges() const noexcept {
    size_t count = 0;
    for (const auto& o : edge_owner_) {
      if (o == Ownership::Shared) ++count;
    }
    return count;
  }
  size_t n_owned_edges() const noexcept {
    size_t count = 0;
    for (const auto& o : edge_owner_) {
      if (o == Ownership::Owned) ++count;
    }
    return count;
  }

  std::vector<index_t> owned_vertices() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_vertices());
    out.reserve(static_cast<size_t>(n));
    for (index_t v = 0; v < n; ++v) {
      if (is_owned_vertex(v)) out.push_back(v);
    }
    return out;
  }
  std::vector<index_t> shared_vertices() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_vertices());
    out.reserve(static_cast<size_t>(n));
    for (index_t v = 0; v < n; ++v) {
      if (is_shared_vertex(v)) out.push_back(v);
    }
    return out;
  }
  std::vector<index_t> ghost_vertices() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_vertices());
    out.reserve(static_cast<size_t>(n));
    for (index_t v = 0; v < n; ++v) {
      if (is_ghost_vertex(v)) out.push_back(v);
    }
    return out;
  }

  std::vector<index_t> owned_cells() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_cells());
    out.reserve(static_cast<size_t>(n));
    for (index_t c = 0; c < n; ++c) {
      if (is_owned_cell(c)) out.push_back(c);
    }
    return out;
  }
  std::vector<index_t> shared_cells() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_cells());
    out.reserve(static_cast<size_t>(n));
    for (index_t c = 0; c < n; ++c) {
      if (is_shared_cell(c)) out.push_back(c);
    }
    return out;
  }
  std::vector<index_t> ghost_cells() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_cells());
    out.reserve(static_cast<size_t>(n));
    for (index_t c = 0; c < n; ++c) {
      if (is_ghost_cell(c)) out.push_back(c);
    }
    return out;
  }

  std::vector<index_t> owned_faces() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_faces());
    out.reserve(static_cast<size_t>(n));
    for (index_t f = 0; f < n; ++f) {
      if (is_owned_face(f)) out.push_back(f);
    }
    return out;
  }
  std::vector<index_t> shared_faces() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_faces());
    out.reserve(static_cast<size_t>(n));
    for (index_t f = 0; f < n; ++f) {
      if (is_shared_face(f)) out.push_back(f);
    }
    return out;
  }
  std::vector<index_t> ghost_faces() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_faces());
    out.reserve(static_cast<size_t>(n));
    for (index_t f = 0; f < n; ++f) {
      if (is_ghost_face(f)) out.push_back(f);
    }
    return out;
  }

  std::vector<index_t> owned_edges() const {
    std::vector<index_t> out;
    out.reserve(edge_owner_.size());
    for (index_t e = 0; e < static_cast<index_t>(edge_owner_.size()); ++e) {
      if (edge_owner_[static_cast<size_t>(e)] == Ownership::Owned) out.push_back(e);
    }
    return out;
  }
  std::vector<index_t> shared_edges() const {
    std::vector<index_t> out;
    out.reserve(edge_owner_.size());
    for (index_t e = 0; e < static_cast<index_t>(edge_owner_.size()); ++e) {
      if (edge_owner_[static_cast<size_t>(e)] == Ownership::Shared) out.push_back(e);
    }
    return out;
  }
  std::vector<index_t> ghost_edges() const {
    std::vector<index_t> out;
    out.reserve(edge_owner_.size());
    for (index_t e = 0; e < static_cast<index_t>(edge_owner_.size()); ++e) {
      if (edge_owner_[static_cast<size_t>(e)] == Ownership::Ghost) out.push_back(e);
    }
    return out;
  }

	  void set_ownership(index_t id, EntityKind kind, Ownership own, rank_t owner_rank = -1) {
	    auto& mesh = local_mesh();
	    switch (kind) {
	      case EntityKind::Volume:
	        ensure_size(cell_owner_, mesh.n_cells());
	        ensure_size(cell_owner_rank_, mesh.n_cells(), 0);
	        cell_owner_[id] = own;
	        cell_owner_rank_[id] = owner_rank >= 0 ? owner_rank : 0;
	        break;
	      case EntityKind::Vertex:
	        ensure_size(vertex_owner_, mesh.n_vertices());
	        ensure_size(vertex_owner_rank_, mesh.n_vertices(), 0);
	        vertex_owner_[id] = own;
	        vertex_owner_rank_[id] = owner_rank >= 0 ? owner_rank : 0;
	        break;
	      case EntityKind::Face:
	        ensure_size(face_owner_, mesh.n_faces());
	        ensure_size(face_owner_rank_, mesh.n_faces(), 0);
	        face_owner_[id] = own;
	        face_owner_rank_[id] = owner_rank >= 0 ? owner_rank : 0;
	        break;
	      case EntityKind::Edge:
	        ensure_size(edge_owner_, mesh.n_edges());
	        ensure_size(edge_owner_rank_, mesh.n_edges(), 0);
	        edge_owner_[id] = own;
	        edge_owner_rank_[id] = owner_rank >= 0 ? owner_rank : 0;
	        break;
	    }
	    mesh.event_bus().notify(MeshEvent::PartitionChanged);
	  }

	  // Ghosts (no-op in serial stub)
	  void build_ghost_layer(int) { local_mesh().event_bus().notify(MeshEvent::PartitionChanged); }
	  void clear_ghosts() { local_mesh().event_bus().notify(MeshEvent::PartitionChanged); }
	  void update_ghosts(const std::vector<FieldHandle>&) { local_mesh().event_bus().notify(MeshEvent::FieldsChanged); }
	  void update_exchange_ghost_fields() { local_mesh().event_bus().notify(MeshEvent::FieldsChanged); }
	  void update_exchange_ghost_coordinates(Configuration cfg = Configuration::Current) {
	    const bool use_current = (cfg == Configuration::Current || cfg == Configuration::Deformed);
	    if (!use_current) return;
	    if (!local_mesh().has_current_coords()) {
	      local_mesh().set_current_coords(local_mesh().X_ref());
	    }
	    local_mesh().event_bus().notify(MeshEvent::GeometryChanged);
	  }

	  // Migration & balancing (no-op)
	  void migrate(const std::vector<rank_t>&) { local_mesh().event_bus().notify(MeshEvent::PartitionChanged); }
	  void rebalance(PartitionHint, const std::unordered_map<std::string,std::string>& = {}) {
	    local_mesh().event_bus().notify(MeshEvent::PartitionChanged);
	  }

  // Partition metrics (single-rank computation)
  struct PartitionMetrics {
    double load_imbalance_factor{0.0};
    size_t min_cells_per_rank{0};
    size_t max_cells_per_rank{0};
    size_t avg_cells_per_rank{0};
    size_t total_edge_cuts{0};
    size_t total_shared_faces{0};
    size_t total_ghost_cells{0};
    double avg_neighbors_per_rank{0.0};
    size_t min_memory_per_rank{0};
    size_t max_memory_per_rank{0};
    double memory_imbalance_factor{0.0};
    size_t cells_to_migrate{0};
    size_t migration_volume{0};
  };

	  PartitionMetrics compute_partition_quality() const {
	    PartitionMetrics m;
	    size_t cells = local_mesh().n_cells();
	    m.min_cells_per_rank = m.max_cells_per_rank = m.avg_cells_per_rank = cells;
	    return m;
	  }

  // Parallel I/O stubs
	  static DistributedMesh load_parallel(const MeshIOOptions& opts, MeshComm = MeshComm::world()) {
	    DistributedMesh dm;
	    dm.local_mesh() = MeshBase::load(opts);
	    return dm;
	  }
	  void save_parallel(const MeshIOOptions& opts) const { local_mesh().save(opts); }

	  // Global reductions (single rank)
	  size_t global_n_vertices() const { return local_mesh().n_vertices(); }
	  size_t global_n_cells() const { return local_mesh().n_cells(); }
	  size_t global_n_faces() const { return local_mesh().n_faces(); }
	  BoundingBox global_bounding_box() const { return local_mesh().bounding_box(); }

	  // Distributed search (serial)
	  PointLocateResult locate_point_global(const std::array<real_t,3>& x,
	                                        Configuration cfg = Configuration::Reference) const {
	    return local_mesh().locate_point(x, cfg);
	  }

	  std::vector<PointLocateResult> locate_points_global(
	      const std::vector<std::array<real_t,3>>& X,
	      Configuration cfg = Configuration::Reference) const {
	    return local_mesh().locate_points(X, cfg);
	  }

  // Exchange patterns
  struct ExchangePattern {
    std::vector<rank_t> send_ranks;
    std::vector<std::vector<index_t>> send_lists;
    std::vector<rank_t> recv_ranks;
    std::vector<std::vector<index_t>> recv_lists;
  };
  const ExchangePattern& vertex_exchange_pattern() const { return vertex_exchange_; }
  const ExchangePattern& cell_exchange_pattern() const { return cell_exchange_; }
  const ExchangePattern& face_exchange_pattern() const { return face_exchange_; }
  const ExchangePattern& edge_exchange_pattern() const { return edge_exchange_; }
  void build_exchange_patterns() { /* no-op, patterns empty */ }

private:
  template<typename T>
  static void ensure_size(std::vector<T>& v, size_t n, const T& value = T()) {
    if (v.size() < n) v.resize(n, value);
  }
  static Ownership get_owner(const std::vector<Ownership>& v, Ownership def, index_t i) {
    if (i < 0) return def;
    size_t n = v.size();
    return (static_cast<size_t>(i) < n ? v[static_cast<size_t>(i)] : def);
  }
  static rank_t get_owner_rank(const std::vector<rank_t>& v, index_t i) {
    if (i < 0) return 0;
    size_t n = v.size();
    return (static_cast<size_t>(i) < n ? v[static_cast<size_t>(i)] : 0);
  }

  std::shared_ptr<MeshBase> local_mesh_;
  std::unordered_set<rank_t> neighbor_ranks_;
  std::vector<Ownership> vertex_owner_, edge_owner_, face_owner_, cell_owner_;
  std::vector<rank_t> cell_owner_rank_, face_owner_rank_, vertex_owner_rank_, edge_owner_rank_;
  ExchangePattern vertex_exchange_, cell_exchange_;
  ExchangePattern face_exchange_, edge_exchange_;

  void invalidate_exchange_patterns_() {
    neighbor_ranks_.clear();
    vertex_exchange_ = ExchangePattern{};
    cell_exchange_ = ExchangePattern{};
    face_exchange_ = ExchangePattern{};
    edge_exchange_ = ExchangePattern{};
  }

  void reset_partition_state_() {
    invalidate_exchange_patterns_();
    vertex_owner_.assign(local_mesh().n_vertices(), Ownership::Owned);
    vertex_owner_rank_.assign(local_mesh().n_vertices(), 0);
    edge_owner_.assign(local_mesh().n_edges(), Ownership::Owned);
    edge_owner_rank_.assign(local_mesh().n_edges(), 0);
    cell_owner_.assign(local_mesh().n_cells(), Ownership::Owned);
    cell_owner_rank_.assign(local_mesh().n_cells(), 0);
    face_owner_.assign(local_mesh().n_faces(), Ownership::Owned);
    face_owner_rank_.assign(local_mesh().n_faces(), 0);
    local_mesh().event_bus().notify(MeshEvent::PartitionChanged);
  }
};

// ========================
// Template specialization for compile-time dimension (serial stub)
// ========================
template <int Dim>
class DistributedMesh_t {
public:
  explicit DistributedMesh_t(std::shared_ptr<DistributedMesh> dmesh)
    : dmesh_(std::move(dmesh))
  {
    if (!dmesh_) throw std::invalid_argument("DistributedMesh_t: null distributed mesh");
    if (dmesh_->local_mesh().dim() != 0 && dmesh_->local_mesh().dim() != Dim) {
      throw std::invalid_argument("DistributedMesh_t: dimension mismatch");
    }
  }

  int dim() const noexcept { return Dim; }
  DistributedMesh& dist_mesh() { return *dmesh_; }
  const DistributedMesh& dist_mesh() const { return *dmesh_; }

private:
  std::shared_ptr<DistributedMesh> dmesh_;
};

#else

// ========================
// Distributed mesh wrapper
// ========================
class DistributedMesh {
public:
  // ---- Constructors
  DistributedMesh();
  explicit DistributedMesh(MPI_Comm comm);
  explicit DistributedMesh(std::shared_ptr<MeshBase> local_mesh, MPI_Comm comm = MPI_COMM_WORLD);
  explicit DistributedMesh(MeshComm comm) : DistributedMesh(comm.native()) {}
  explicit DistributedMesh(std::shared_ptr<MeshBase> local_mesh, MeshComm comm)
    : DistributedMesh(std::move(local_mesh), comm.native()) {}

  // ---- Access to underlying mesh
  MeshBase& local_mesh() { return *local_mesh_; }
  const MeshBase& local_mesh() const { return *local_mesh_; }
  std::shared_ptr<MeshBase> local_mesh_ptr() { return local_mesh_; }
  std::shared_ptr<const MeshBase> local_mesh_ptr() const { return local_mesh_; }

  // ---- MeshBase-like surface (local meaning)
  using LoadFn = MeshBase::LoadFn;
  using SaveFn = MeshBase::SaveFn;

  MeshBase& base() { return local_mesh(); }
  const MeshBase& base() const { return local_mesh(); }

  // Lifecycle/builders
  void clear() {
    local_mesh().clear();
    reset_partition_state_();
  }
  void reserve(index_t n_vertices, index_t n_cells, index_t n_faces = 0) {
    local_mesh().reserve(n_vertices, n_cells, n_faces);
  }
  // Builders
  // - `build_from_arrays(...)` is a *local* build in all builds (serial + MPI).
  // - Use `build_from_arrays_global_and_partition(...)` to build from a global mesh
  //   on rank 0 and distribute/partition it across ranks.
  void build_from_arrays_local(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape) {
    local_mesh().build_from_arrays(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
    reset_partition_state_();
  }
  void build_from_arrays(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape) {
    build_from_arrays_local(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
  }

  void build_from_arrays_global_and_partition(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape,
      PartitionHint hint = PartitionHint::Cells,
      int ghost_layers = 0,
      const std::unordered_map<std::string, std::string>& options = {});
  void set_faces_from_arrays(
      const std::vector<CellShape>& face_shape,
      const std::vector<offset_t>& face2vertex_offsets,
      const std::vector<index_t>& face2vertex,
      const std::vector<std::array<index_t,2>>& face2cell) {
    local_mesh().set_faces_from_arrays(face_shape, face2vertex_offsets, face2vertex, face2cell);
    reset_partition_state_();
  }
  void set_cell_faces_from_arrays(
      const std::vector<offset_t>& cell2face_offsets,
      const std::vector<index_t>& cell2face,
      const std::vector<int8_t>& cell2face_sense) {
    local_mesh().set_cell_faces_from_arrays(cell2face_offsets, cell2face, cell2face_sense);
    reset_partition_state_();
  }
  void set_cell_faces_from_arrays(
      const std::vector<offset_t>& cell2face_offsets,
      const std::vector<index_t>& cell2face,
      const std::vector<int8_t>& cell2face_sense,
      const std::vector<int8_t>& cell2face_perm) {
    local_mesh().set_cell_faces_from_arrays(cell2face_offsets, cell2face, cell2face_sense, cell2face_perm);
    reset_partition_state_();
  }
  void set_edges_from_arrays(const std::vector<std::array<index_t,2>>& edge2vertex) {
    local_mesh().set_edges_from_arrays(edge2vertex);
    reset_partition_state_();
  }
  void finalize() {
    local_mesh().finalize();
    reset_partition_state_();
  }

  // Basic queries
  int dim() const noexcept { return local_mesh().dim(); }
  const std::string& mesh_id() const noexcept { return local_mesh().mesh_id(); }
  size_t n_vertices() const noexcept { return local_mesh().n_vertices(); }
  size_t n_cells() const noexcept { return local_mesh().n_cells(); }
  size_t n_faces() const noexcept { return local_mesh().n_faces(); }
  size_t n_edges() const noexcept { return local_mesh().n_edges(); }
  size_t n_boundary_faces() const { return local_mesh().n_boundary_faces(); }

  // Coordinates
  const std::vector<real_t>& X_ref() const noexcept { return local_mesh().X_ref(); }
  const std::vector<real_t>& X_cur() const noexcept { return local_mesh().X_cur(); }
  bool has_current_coords() const noexcept { return local_mesh().has_current_coords(); }
  real_t* X_cur_data_mutable() noexcept { return local_mesh().X_cur_data_mutable(); }
  void set_current_coords(const std::vector<real_t>& Xcur) { local_mesh().set_current_coords(Xcur); }
  void clear_current_coords() { local_mesh().clear_current_coords(); }
  Configuration active_configuration() const noexcept { return local_mesh().active_configuration(); }
  void use_reference_configuration() { local_mesh().use_reference_configuration(); }
  void use_current_configuration() { local_mesh().use_current_configuration(); }
  std::array<real_t,3> get_vertex_coords(index_t v) const { return local_mesh().get_vertex_coords(v); }
  void set_vertex_coords(index_t v, const std::array<real_t,3>& xyz) { local_mesh().set_vertex_coords(v, xyz); }
  void set_vertex_deformed_coords(index_t v, const std::array<real_t,3>& xyz) {
    local_mesh().set_vertex_deformed_coords(v, xyz);
  }

  // Topology access
  const std::vector<CellShape>& cell_shapes() const noexcept { return local_mesh().cell_shapes(); }
  const CellShape& cell_shape(index_t c) const { return local_mesh().cell_shape(c); }
  std::pair<const index_t*, size_t> cell_vertices_span(index_t c) const { return local_mesh().cell_vertices_span(c); }
  std::vector<index_t> cell_vertices(index_t c) const { return local_mesh().cell_vertices(c); }
  std::vector<index_t> get_cell_connectivity(index_t c) const { return local_mesh().get_cell_connectivity(c); }
  const std::vector<offset_t>& cell2vertex_offsets() const noexcept { return local_mesh().cell2vertex_offsets(); }
  const std::vector<index_t>& cell2vertex() const noexcept { return local_mesh().cell2vertex(); }

  const std::vector<CellShape>& face_shapes() const noexcept { return local_mesh().face_shapes(); }
  std::pair<const index_t*, size_t> face_vertices_span(index_t f) const { return local_mesh().face_vertices_span(f); }
  std::vector<index_t> face_vertices(index_t f) const { return local_mesh().face_vertices(f); }
  std::vector<index_t> cell_faces(index_t c) const { return local_mesh().cell_faces(c); }
  std::pair<const index_t*, size_t> cell_faces_span(index_t c) const { return local_mesh().cell_faces_span(c); }
  std::pair<const int8_t*, size_t> cell_face_senses_span(index_t c) const {
    return local_mesh().cell_face_senses_span(c);
  }
  std::pair<const int8_t*, size_t> cell_face_permutations_span(index_t c) const {
    return local_mesh().cell_face_permutations_span(c);
  }
  const std::vector<offset_t>& face2vertex_offsets() const noexcept { return local_mesh().face2vertex_offsets(); }
  const std::vector<index_t>& face2vertex() const noexcept { return local_mesh().face2vertex(); }
  const std::vector<std::array<index_t,2>>& face2cell() const noexcept { return local_mesh().face2cell(); }
  const std::vector<label_t>& face_boundary_ids() const noexcept { return local_mesh().face_boundary_ids(); }

  // Incremental builders
  void add_vertex(index_t id, const std::array<real_t,3>& xyz) {
    local_mesh().add_vertex(id, xyz);
    reset_partition_state_();
  }
  void add_cell(index_t id, CellFamily family, const std::vector<index_t>& vertices) {
    local_mesh().add_cell(id, family, vertices);
    reset_partition_state_();
  }
  void add_boundary_face(index_t id, const std::vector<index_t>& vertices) {
    local_mesh().add_boundary_face(id, vertices);
    reset_partition_state_();
  }
  const std::array<index_t,2>& face_cells(index_t f) const { return local_mesh().face_cells(f); }
  const std::vector<std::array<index_t,2>>& edge2vertex() const noexcept { return local_mesh().edge2vertex(); }
  const std::array<index_t,2>& edge_vertices(index_t e) const { return local_mesh().edge_vertices(e); }

  // IDs / ownership
  const std::vector<gid_t>& vertex_gids() const noexcept { return local_mesh().vertex_gids(); }
  const std::vector<gid_t>& cell_gids() const noexcept { return local_mesh().cell_gids(); }
  const std::vector<gid_t>& face_gids() const noexcept { return local_mesh().face_gids(); }
  const std::vector<gid_t>& edge_gids() const noexcept { return local_mesh().edge_gids(); }
  void set_vertex_gids(std::vector<gid_t> gids) {
    local_mesh().set_vertex_gids(std::move(gids));
    invalidate_exchange_patterns_();
  }
  void set_cell_gids(std::vector<gid_t> gids) {
    local_mesh().set_cell_gids(std::move(gids));
    invalidate_exchange_patterns_();
  }
  void set_face_gids(std::vector<gid_t> gids) {
    local_mesh().set_face_gids(std::move(gids));
    invalidate_exchange_patterns_();
  }
  void set_edge_gids(std::vector<gid_t> gids) { local_mesh().set_edge_gids(std::move(gids)); }
  index_t global_to_local_cell(gid_t gid) const { return local_mesh().global_to_local_cell(gid); }
  index_t global_to_local_vertex(gid_t gid) const { return local_mesh().global_to_local_vertex(gid); }
  index_t global_to_local_face(gid_t gid) const { return local_mesh().global_to_local_face(gid); }
  index_t global_to_local_edge(gid_t gid) const { return local_mesh().global_to_local_edge(gid); }

  // Adaptivity metadata
  size_t refinement_level(index_t cell) const { return local_mesh().refinement_level(cell); }
  void set_refinement_level(index_t cell, size_t level) { local_mesh().set_refinement_level(cell, level); }
  const std::vector<size_t>& cell_refinement_levels() const noexcept { return local_mesh().cell_refinement_levels(); }
  void set_cell_refinement_levels(std::vector<size_t> levels) {
    local_mesh().set_cell_refinement_levels(std::move(levels));
  }

  // Labels & sets
  void set_region_label(index_t cell, label_t label) { local_mesh().set_region_label(cell, label); }
  label_t region_label(index_t cell) const { return local_mesh().region_label(cell); }
  std::vector<index_t> cells_with_label(label_t label) const { return local_mesh().cells_with_label(label); }
  const std::vector<label_t>& cell_region_ids() const noexcept { return local_mesh().cell_region_ids(); }

  void set_boundary_label(index_t face, label_t label) { local_mesh().set_boundary_label(face, label); }
  label_t boundary_label(index_t face) const { return local_mesh().boundary_label(face); }
  std::vector<index_t> faces_with_label(label_t label) const { return local_mesh().faces_with_label(label); }

  void set_edge_label(index_t edge, label_t label) { local_mesh().set_edge_label(edge, label); }
  label_t edge_label(index_t edge) const { return local_mesh().edge_label(edge); }
  std::vector<index_t> edges_with_label(label_t label) const { return local_mesh().edges_with_label(label); }
  const std::vector<label_t>& edge_label_ids() const noexcept { return local_mesh().edge_label_ids(); }

  void set_vertex_label(index_t vertex, label_t label) { local_mesh().set_vertex_label(vertex, label); }
  label_t vertex_label(index_t vertex) const { return local_mesh().vertex_label(vertex); }
  std::vector<index_t> vertices_with_label(label_t label) const { return local_mesh().vertices_with_label(label); }
  const std::vector<label_t>& vertex_label_ids() const noexcept { return local_mesh().vertex_label_ids(); }

  void add_to_set(EntityKind kind, const std::string& name, index_t id) { local_mesh().add_to_set(kind, name, id); }
  const std::vector<index_t>& get_set(EntityKind kind, const std::string& name) const {
    return local_mesh().get_set(kind, name);
  }
  bool has_set(EntityKind kind, const std::string& name) const { return local_mesh().has_set(kind, name); }
  void remove_from_set(EntityKind kind, const std::string& name, index_t id) {
    local_mesh().remove_from_set(kind, name, id);
  }
  void remove_set(EntityKind kind, const std::string& name) { local_mesh().remove_set(kind, name); }
  std::vector<std::string> list_sets(EntityKind kind) const { return local_mesh().list_sets(kind); }

  void register_label(const std::string& name, label_t label) { local_mesh().register_label(name, label); }
  std::string label_name(label_t label) const { return local_mesh().label_name(label); }
  label_t label_from_name(const std::string& name) const { return local_mesh().label_from_name(name); }
  std::unordered_map<label_t, std::string> list_label_names() const { return local_mesh().list_label_names(); }
  void clear_label_registry() { local_mesh().clear_label_registry(); }

  // Fields
  FieldHandle attach_field(EntityKind kind, const std::string& name, FieldScalarType type,
                           size_t components, size_t custom_bytes_per_component = 0) {
    return local_mesh().attach_field(kind, name, type, components, custom_bytes_per_component);
  }
  FieldHandle attach_field_with_descriptor(EntityKind kind, const std::string& name,
                                           FieldScalarType type, const FieldDescriptor& descriptor) {
    return local_mesh().attach_field_with_descriptor(kind, name, type, descriptor);
  }
  bool has_field(EntityKind kind, const std::string& name) const { return local_mesh().has_field(kind, name); }
  FieldHandle field_handle(EntityKind kind, const std::string& name) const { return local_mesh().field_handle(kind, name); }
  void remove_field(const FieldHandle& h) { local_mesh().remove_field(h); }
  void* field_data(const FieldHandle& h) { return local_mesh().field_data(h); }
  const void* field_data(const FieldHandle& h) const { return local_mesh().field_data(h); }
  size_t field_components(const FieldHandle& h) const { return local_mesh().field_components(h); }
  FieldScalarType field_type(const FieldHandle& h) const { return local_mesh().field_type(h); }
  size_t field_entity_count(const FieldHandle& h) const { return local_mesh().field_entity_count(h); }
  size_t field_bytes_per_entity(const FieldHandle& h) const { return local_mesh().field_bytes_per_entity(h); }
  const FieldDescriptor* field_descriptor(const FieldHandle& h) const { return local_mesh().field_descriptor(h); }
  void set_field_descriptor(const FieldHandle& h, const FieldDescriptor& descriptor) {
    local_mesh().set_field_descriptor(h, descriptor);
  }
  void resize_fields(EntityKind kind, size_t new_count) { local_mesh().resize_fields(kind, new_count); }
  std::vector<std::string> field_names(EntityKind kind) const { return local_mesh().field_names(kind); }
  void* field_data_by_name(EntityKind kind, const std::string& name) { return local_mesh().field_data_by_name(kind, name); }
  const void* field_data_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_data_by_name(kind, name);
  }
  size_t field_components_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_components_by_name(kind, name);
  }
  FieldScalarType field_type_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_type_by_name(kind, name);
  }
  size_t field_bytes_per_component_by_name(EntityKind kind, const std::string& name) const {
    return local_mesh().field_bytes_per_component_by_name(kind, name);
  }
  template <typename T>
  T* field_data_as(const FieldHandle& h) { return local_mesh().field_data_as<T>(h); }
  template <typename T>
  const T* field_data_as(const FieldHandle& h) const { return local_mesh().field_data_as<T>(h); }

  // Geometry operations
  std::array<real_t,3> cell_center(index_t c, Configuration cfg = Configuration::Reference) const {
    return local_mesh().cell_center(c, cfg);
  }
  std::array<real_t,3> cell_centroid(index_t c, Configuration cfg = Configuration::Reference) const {
    return local_mesh().cell_centroid(c, cfg);
  }
  std::array<real_t,3> face_center(index_t f, Configuration cfg = Configuration::Reference) const {
    return local_mesh().face_center(f, cfg);
  }
  std::array<real_t,3> face_normal(index_t f, Configuration cfg = Configuration::Reference) const {
    return local_mesh().face_normal(f, cfg);
  }
  real_t face_area(index_t f, Configuration cfg = Configuration::Reference) const { return local_mesh().face_area(f, cfg); }
  real_t cell_measure(index_t c, Configuration cfg = Configuration::Reference) const { return local_mesh().cell_measure(c, cfg); }
  BoundingBox bounding_box(Configuration cfg = Configuration::Reference) const { return local_mesh().bounding_box(cfg); }

  // Quality metrics
  real_t compute_quality(index_t cell, const std::string& metric = "aspect_ratio") const {
    return local_mesh().compute_quality(cell, metric);
  }
  std::pair<real_t,real_t> global_quality_range(const std::string& metric = "aspect_ratio") const;

  // Adjacency queries
  std::vector<index_t> cell_neighbors(index_t c) const { return local_mesh().cell_neighbors(c); }
  std::vector<index_t> vertex_cells(index_t v) const { return local_mesh().vertex_cells(v); }
  std::vector<index_t> face_neighbors(index_t f) const { return local_mesh().face_neighbors(f); }
  std::vector<index_t> boundary_faces() const { return local_mesh().boundary_faces(); }
  std::vector<index_t> boundary_cells() const { return local_mesh().boundary_cells(); }
  void build_vertex2cell() { local_mesh().build_vertex2cell(); }
  void build_vertex2face() { local_mesh().build_vertex2face(); }
  void build_cell2cell() { local_mesh().build_cell2cell(); }

  // Submesh extraction
  MeshBase extract_submesh_by_region(label_t region_label) const { return local_mesh().extract_submesh_by_region(region_label); }
  MeshBase extract_submesh_by_regions(const std::vector<label_t>& region_labels) const {
    return local_mesh().extract_submesh_by_regions(region_labels);
  }
  MeshBase extract_submesh_by_boundary(label_t boundary_label) const { return local_mesh().extract_submesh_by_boundary(boundary_label); }

  // Search & point location
  PointLocateResult locate_point(const std::array<real_t,3>& x, Configuration cfg = Configuration::Reference) const {
    return local_mesh().locate_point(x, cfg);
  }
  std::vector<PointLocateResult> locate_points(const std::vector<std::array<real_t,3>>& X,
                                               Configuration cfg = Configuration::Reference) const {
    return local_mesh().locate_points(X, cfg);
  }
  RayIntersectResult intersect_ray(const std::array<real_t,3>& origin,
                                   const std::array<real_t,3>& direction,
                                   Configuration cfg = Configuration::Reference) const {
    return local_mesh().intersect_ray(origin, direction, cfg);
  }
  void build_search_structure(Configuration cfg = Configuration::Reference) const { local_mesh().build_search_structure(cfg); }
  void build_search_structure(const MeshSearch::SearchConfig& config,
                              Configuration cfg = Configuration::Reference) const {
    local_mesh().build_search_structure(config, cfg);
  }
  void clear_search_structure() const { local_mesh().clear_search_structure(); }
  bool has_search_structure() const { return local_mesh().has_search_structure(); }
  const SearchAccel* search_accel() const noexcept { return local_mesh().search_accel(); }

  // Validation & diagnostics
  void validate_basic() const { local_mesh().validate_basic(); }
  void validate_topology() const { local_mesh().validate_topology(); }
  void validate_geometry() const { local_mesh().validate_geometry(); }
  void report_statistics() const { local_mesh().report_statistics(); }
  void write_debug(const std::string& prefix, const std::string& format = "vtu") const {
    local_mesh().write_debug(prefix, format);
  }

  // Memory management
  void shrink_to_fit() { local_mesh().shrink_to_fit(); }
  size_t memory_usage_bytes() const { return local_mesh().memory_usage_bytes(); }

  // Event system
  MeshEventBus& event_bus() { return local_mesh().event_bus(); }
  MeshEventBus& event_bus() const { return local_mesh().event_bus(); }

  // I/O (local meaning)
	  void save(const MeshIOOptions& opts) const { local_mesh().save(opts); }

	  // ---- MPI info
	  MPI_Comm mpi_comm() const noexcept { return user_comm_; }
	  rank_t rank() const noexcept { return my_rank_; }
	  int world_size() const noexcept { return world_size_; }
	  const std::unordered_set<rank_t>& neighbor_ranks() const noexcept { return neighbor_ranks_; }

	  void set_mpi_comm(MPI_Comm comm);

  // ---- Ownership & ghosting
  bool is_owned_cell(index_t i) const;
  bool is_ghost_cell(index_t i) const;
  bool is_shared_cell(index_t i) const;
  rank_t owner_rank_cell(index_t i) const;

  bool is_owned_vertex(index_t i) const;
  bool is_ghost_vertex(index_t i) const;
  bool is_shared_vertex(index_t i) const;
  rank_t owner_rank_vertex(index_t i) const;

  bool is_owned_face(index_t i) const;
  bool is_ghost_face(index_t i) const;
  bool is_shared_face(index_t i) const;
  rank_t owner_rank_face(index_t i) const;

  bool is_owned_edge(index_t i) const;
  bool is_ghost_edge(index_t i) const;
  bool is_shared_edge(index_t i) const;
  rank_t owner_rank_edge(index_t i) const;

  // ---- Explicit distributed semantics (Phase 3)
  // Default `n_*()` methods on DistributedMesh refer to the *local* mesh size,
  // which includes ghosts when a ghost layer is present. Use these helpers for
  // unambiguous owned/shared/ghost counts and iteration.
  size_t n_owned_vertices() const noexcept {
    size_t count = 0;
    for (const auto& o : vertex_owner_) {
      if (o == Ownership::Owned) ++count;
    }
    return count;
  }
  size_t n_shared_vertices() const noexcept {
    size_t count = 0;
    for (const auto& o : vertex_owner_) {
      if (o == Ownership::Shared) ++count;
    }
    return count;
  }
  size_t n_ghost_vertices() const noexcept {
    size_t count = 0;
    for (const auto& o : vertex_owner_) {
      if (o == Ownership::Ghost) ++count;
    }
    return count;
  }

  size_t n_owned_cells() const noexcept {
    size_t count = 0;
    for (const auto& o : cell_owner_) {
      if (o == Ownership::Owned) ++count;
    }
    return count;
  }
  size_t n_shared_cells() const noexcept {
    size_t count = 0;
    for (const auto& o : cell_owner_) {
      if (o == Ownership::Shared) ++count;
    }
    return count;
  }
  size_t n_ghost_cells() const noexcept {
    size_t count = 0;
    for (const auto& o : cell_owner_) {
      if (o == Ownership::Ghost) ++count;
    }
    return count;
  }

  size_t n_owned_faces() const noexcept {
    size_t count = 0;
    for (const auto& o : face_owner_) {
      if (o == Ownership::Owned) ++count;
    }
    return count;
  }
  size_t n_shared_faces() const noexcept {
    size_t count = 0;
    for (const auto& o : face_owner_) {
      if (o == Ownership::Shared) ++count;
    }
    return count;
  }
  size_t n_ghost_faces() const noexcept {
    size_t count = 0;
    for (const auto& o : face_owner_) {
      if (o == Ownership::Ghost) ++count;
    }
    return count;
  }

  // Edge ownership is not tracked explicitly yet; provide a stable, local-only classification
  // derived from endpoint vertex ownership (ghost > shared > owned).
  size_t n_ghost_edges() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_edges());
    for (index_t e = 0; e < n; ++e) {
      const auto ev = local_mesh().edge_vertices(e);
      if (is_ghost_vertex(ev[0]) || is_ghost_vertex(ev[1])) ++count;
    }
    return count;
  }
  size_t n_shared_edges() const noexcept {
    size_t count = 0;
    const index_t n = static_cast<index_t>(local_mesh().n_edges());
    for (index_t e = 0; e < n; ++e) {
      const auto ev = local_mesh().edge_vertices(e);
      if (is_ghost_vertex(ev[0]) || is_ghost_vertex(ev[1])) continue;
      if (is_shared_vertex(ev[0]) || is_shared_vertex(ev[1])) ++count;
    }
    return count;
  }
  size_t n_owned_edges() const noexcept {
    const size_t total = local_mesh().n_edges();
    return total - n_shared_edges() - n_ghost_edges();
  }

  std::vector<index_t> owned_vertices() const {
    std::vector<index_t> out;
    out.reserve(vertex_owner_.size());
    for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
      if (vertex_owner_[static_cast<size_t>(v)] == Ownership::Owned) out.push_back(v);
    }
    return out;
  }
  std::vector<index_t> shared_vertices() const {
    std::vector<index_t> out;
    out.reserve(vertex_owner_.size());
    for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
      if (vertex_owner_[static_cast<size_t>(v)] == Ownership::Shared) out.push_back(v);
    }
    return out;
  }
  std::vector<index_t> ghost_vertices() const {
    std::vector<index_t> out;
    out.reserve(vertex_owner_.size());
    for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
      if (vertex_owner_[static_cast<size_t>(v)] == Ownership::Ghost) out.push_back(v);
    }
    return out;
  }

  std::vector<index_t> owned_cells() const {
    std::vector<index_t> out;
    out.reserve(cell_owner_.size());
    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
      if (cell_owner_[static_cast<size_t>(c)] == Ownership::Owned) out.push_back(c);
    }
    return out;
  }
  std::vector<index_t> shared_cells() const {
    std::vector<index_t> out;
    out.reserve(cell_owner_.size());
    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
      if (cell_owner_[static_cast<size_t>(c)] == Ownership::Shared) out.push_back(c);
    }
    return out;
  }
  std::vector<index_t> ghost_cells() const {
    std::vector<index_t> out;
    out.reserve(cell_owner_.size());
    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
      if (cell_owner_[static_cast<size_t>(c)] == Ownership::Ghost) out.push_back(c);
    }
    return out;
  }

  std::vector<index_t> owned_faces() const {
    std::vector<index_t> out;
    out.reserve(face_owner_.size());
    for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
      if (face_owner_[static_cast<size_t>(f)] == Ownership::Owned) out.push_back(f);
    }
    return out;
  }
  std::vector<index_t> shared_faces() const {
    std::vector<index_t> out;
    out.reserve(face_owner_.size());
    for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
      if (face_owner_[static_cast<size_t>(f)] == Ownership::Shared) out.push_back(f);
    }
    return out;
  }
  std::vector<index_t> ghost_faces() const {
    std::vector<index_t> out;
    out.reserve(face_owner_.size());
    for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
      if (face_owner_[static_cast<size_t>(f)] == Ownership::Ghost) out.push_back(f);
    }
    return out;
  }

  std::vector<index_t> owned_edges() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_edges());
    out.reserve(static_cast<size_t>(n));
    for (index_t e = 0; e < n; ++e) {
      const auto ev = local_mesh().edge_vertices(e);
      if (is_ghost_vertex(ev[0]) || is_ghost_vertex(ev[1])) continue;
      if (is_shared_vertex(ev[0]) || is_shared_vertex(ev[1])) continue;
      out.push_back(e);
    }
    return out;
  }
  std::vector<index_t> shared_edges() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_edges());
    out.reserve(static_cast<size_t>(n));
    for (index_t e = 0; e < n; ++e) {
      const auto ev = local_mesh().edge_vertices(e);
      if (is_ghost_vertex(ev[0]) || is_ghost_vertex(ev[1])) continue;
      if (is_shared_vertex(ev[0]) || is_shared_vertex(ev[1])) out.push_back(e);
    }
    return out;
  }
  std::vector<index_t> ghost_edges() const {
    std::vector<index_t> out;
    const index_t n = static_cast<index_t>(local_mesh().n_edges());
    out.reserve(static_cast<size_t>(n));
    for (index_t e = 0; e < n; ++e) {
      const auto ev = local_mesh().edge_vertices(e);
      if (is_ghost_vertex(ev[0]) || is_ghost_vertex(ev[1])) out.push_back(e);
    }
    return out;
  }

  void set_ownership(index_t entity_id, EntityKind kind, Ownership ownership, rank_t owner_rank = -1);

  // ---- Ghost layer construction
  void build_ghost_layer(int levels);
  void clear_ghosts();
  void update_ghosts(const std::vector<FieldHandle>& fields);
  void update_exchange_ghost_fields();
  void update_exchange_ghost_coordinates(Configuration cfg = Configuration::Current);

  // ---- Migration & load balancing
  void migrate(const std::vector<rank_t>& new_owner_rank_per_cell);
  void rebalance(PartitionHint hint, const std::unordered_map<std::string,std::string>& options = {});

  // ---- Partition quality metrics
  struct PartitionMetrics {
    // Load balance metrics
    double load_imbalance_factor;  // Max_load / Avg_load - 1.0
    size_t min_cells_per_rank;
    size_t max_cells_per_rank;
    size_t avg_cells_per_rank;

    // Communication metrics
    size_t total_edge_cuts;        // Number of cell-cell edges crossing ranks
    size_t total_shared_faces;     // Number of faces shared between ranks
    size_t total_ghost_cells;      // Total ghost cells across all ranks
    double avg_neighbors_per_rank; // Average number of neighboring ranks

    // Memory metrics
    size_t min_memory_per_rank;    // Bytes
    size_t max_memory_per_rank;    // Bytes
    double memory_imbalance_factor;

    // Migration metrics (if applicable)
    size_t cells_to_migrate;       // Number of cells that would move
    size_t migration_volume;       // Total bytes to transfer
  };

  PartitionMetrics compute_partition_quality() const;

  // ---- Parallel I/O
  static DistributedMesh load_parallel(const MeshIOOptions& opts, MPI_Comm comm);
  static DistributedMesh load_parallel(const MeshIOOptions& opts, MeshComm comm) {
    return load_parallel(opts, comm.native());
  }
  void save_parallel(const MeshIOOptions& opts) const;

  // ---- Global reductions
  size_t global_n_vertices() const;
  size_t global_n_cells() const;
  size_t global_n_faces() const;

  BoundingBox global_bounding_box() const;

  // ---- Distributed search
  PointLocateResult locate_point_global(const std::array<real_t,3>& x,
                                        Configuration cfg = Configuration::Reference) const;

  std::vector<PointLocateResult> locate_points_global(const std::vector<std::array<real_t,3>>& X,
                                                      Configuration cfg = Configuration::Reference) const;

  // ---- Communication patterns
  struct ExchangePattern {
    std::vector<rank_t> send_ranks;
    std::vector<std::vector<index_t>> send_lists; // entities to send per rank
    std::vector<rank_t> recv_ranks;
    std::vector<std::vector<index_t>> recv_lists; // entities to recv per rank
  };

  const ExchangePattern& vertex_exchange_pattern() const { return vertex_exchange_; }
  const ExchangePattern& cell_exchange_pattern() const { return cell_exchange_; }
  const ExchangePattern& face_exchange_pattern() const { return face_exchange_; }

  void build_exchange_patterns();

	private:
	  // Local mesh (owned)
	  std::shared_ptr<MeshBase> local_mesh_;

	  // MPI communicator and info
	  struct CommHolder {
	    MPI_Comm comm = MPI_COMM_SELF;
	    bool owns = false;
	    ~CommHolder();
	  };

	  MPI_Comm user_comm_ = MPI_COMM_SELF;
	  MPI_Comm comm_ = MPI_COMM_SELF;
	  std::shared_ptr<CommHolder> comm_holder_ = std::make_shared<CommHolder>();
	  rank_t my_rank_ = 0;
	  int world_size_ = 1;
	  std::unordered_set<rank_t> neighbor_ranks_;

  // Per-entity ownership
  std::vector<Ownership> vertex_owner_;
  std::vector<Ownership> face_owner_;
  std::vector<Ownership> cell_owner_;
  std::vector<Ownership> edge_owner_;

  // Owner ranks (for shared/ghost entities)
  std::vector<rank_t> cell_owner_rank_;
  std::vector<rank_t> face_owner_rank_;
  std::vector<rank_t> vertex_owner_rank_;
  std::vector<rank_t> edge_owner_rank_;

  // Communication patterns
  ExchangePattern vertex_exchange_;
  ExchangePattern cell_exchange_;
  ExchangePattern face_exchange_;
  ExchangePattern edge_exchange_;

  // Ghost layer metadata
  int ghost_levels_ = 0;
  std::unordered_set<index_t> ghost_vertices_;
  std::unordered_set<index_t> ghost_cells_;
  std::unordered_set<index_t> ghost_faces_;
  std::unordered_set<index_t> ghost_edges_;

  // Helper methods
  void exchange_entity_data(EntityKind kind, const void* send_data, void* recv_data,
                            size_t bytes_per_entity, const ExchangePattern& pattern);
  void gather_shared_entities();
  void sync_ghost_metadata();
  void synchronize_field_data(EntityKind kind, const std::string& field_name);
  void build_from_arrays_global_and_partition_two_phase_parmetis_(
      int spatial_dim,
      const std::vector<real_t>& X_ref,
      const std::vector<offset_t>& cell2vertex_offsets,
      const std::vector<index_t>& cell2vertex,
      const std::vector<CellShape>& cell_shape,
      PartitionHint hint,
      int ghost_layers,
      const std::unordered_map<std::string, std::string>& options);

  void invalidate_exchange_patterns_() {
    neighbor_ranks_.clear();
    vertex_exchange_ = ExchangePattern{};
    cell_exchange_ = ExchangePattern{};
    face_exchange_ = ExchangePattern{};
    edge_exchange_ = ExchangePattern{};
  }

  void reset_partition_state_() {
    invalidate_exchange_patterns_();
    ghost_levels_ = 0;
    ghost_vertices_.clear();
    ghost_cells_.clear();
    ghost_faces_.clear();
    ghost_edges_.clear();
    vertex_owner_.assign(local_mesh().n_vertices(), Ownership::Owned);
    vertex_owner_rank_.assign(local_mesh().n_vertices(), my_rank_);
    cell_owner_.assign(local_mesh().n_cells(), Ownership::Owned);
    cell_owner_rank_.assign(local_mesh().n_cells(), my_rank_);
    face_owner_.assign(local_mesh().n_faces(), Ownership::Owned);
    face_owner_rank_.assign(local_mesh().n_faces(), my_rank_);
    edge_owner_.assign(local_mesh().n_edges(), Ownership::Owned);
    edge_owner_rank_.assign(local_mesh().n_edges(), my_rank_);
    local_mesh().event_bus().notify(MeshEvent::PartitionChanged);
  }
};

// ========================
// Template specialization for compile-time dimension
// ========================
template <int Dim>
class DistributedMesh_t {
public:
  explicit DistributedMesh_t(std::shared_ptr<DistributedMesh> dmesh)
    : dmesh_(std::move(dmesh))
  {
    if (!dmesh_) throw std::invalid_argument("DistributedMesh_t: null distributed mesh");
    if (dmesh_->local_mesh().dim() != 0 && dmesh_->local_mesh().dim() != Dim) {
      throw std::invalid_argument("DistributedMesh_t: dimension mismatch");
    }
  }

  int dim() const noexcept { return Dim; }
  DistributedMesh& dist_mesh() { return *dmesh_; }
  const DistributedMesh& dist_mesh() const { return *dmesh_; }

  // TODO: Implement when Mesh<Dim> wrapper is available
  // Mesh<Dim> local_mesh() {
  //   return Mesh<Dim>(dmesh_->local_mesh_ptr());
  // }

private:
  std::shared_ptr<DistributedMesh> dmesh_;
};

#if defined(MESH_BUILD_TESTS)
namespace test::internal {
void serialize_mesh_for_test(const MeshBase& mesh, std::vector<char>& buffer);
void deserialize_mesh_for_test(const std::vector<char>& buffer, MeshBase& mesh);
} // namespace test::internal
#endif

#endif // MESH_HAS_MPI

} // namespace svmp

#endif // SVMP_DISTRIBUTED_MESH_H
