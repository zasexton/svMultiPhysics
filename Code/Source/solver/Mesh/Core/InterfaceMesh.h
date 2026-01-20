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

#ifndef SVMP_INTERFACE_MESH_H
#define SVMP_INTERFACE_MESH_H

#include "MeshBase.h"
#include "../Geometry/MeshOrientation.h"
#include "../Topology/CellTopology.h"
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

namespace svmp {

// ====================
// P0 #1: Interface & Coupling Surfaces (Trace/InterfaceMesh)
// ====================
// First-class "surface meshes" derived from face sets, with:
//   - Stable global IDs
//   - Consistent orientation and normals
//   - Incidence back to volume cells
//   - Support for FSI walls, CHT surfaces, contact pairs, mortar/Nitsche interfaces,
//     electrode or endocardial surfaces in EP

class InterfaceMesh {
public:
  // ---- Construction
  InterfaceMesh() = default;

  // Build from a face set in a volume mesh
  static InterfaceMesh build_from_face_set(
      const MeshBase& volume_mesh,
      const std::string& face_set_name,
      bool compute_orientation = true);

  // Build from boundary faces with a specific label
  static InterfaceMesh build_from_boundary_label(
      const MeshBase& volume_mesh,
      label_t boundary_label,
      bool compute_orientation = true);

  // Build from region boundary (extract all faces on the boundary of a region)
  static InterfaceMesh build_from_region_boundary(
      const MeshBase& volume_mesh,
      label_t region_label,
      bool compute_orientation = true);

  // ---- Basic queries
  size_t n_faces() const { return trace_face_gid_.size(); }
  size_t n_vertices() const { return trace_vertex_gid_.size(); }
  int spatial_dim() const { return spatial_dim_; }

  // ---- Topology access
  const std::vector<gid_t>& face_gids() const { return trace_face_gid_; }
  const std::vector<gid_t>& vertex_gids() const { return trace_vertex_gid_; }

  // Face connectivity (CSR format)
  const std::vector<offset_t>& face2vertex_offsets() const { return trace_face2vertex_offsets_; }
  const std::vector<index_t>& face2vertex() const { return trace_face2vertex_; }
  std::pair<const index_t*, size_t> face_vertices_span(index_t local_face_id) const {
    size_t start = static_cast<size_t>(trace_face2vertex_offsets_[local_face_id]);
    size_t end = static_cast<size_t>(trace_face2vertex_offsets_[local_face_id + 1]);
    return {&trace_face2vertex_[start], end - start};
  }

  // ---- Incidence to volume mesh
  // Maps each trace face to its parent volume cell and local face index within that cell
  index_t volume_face(index_t local_face_id) const {
    return trace_face_to_volume_face_.at(static_cast<size_t>(local_face_id));
  }

  index_t volume_cell(index_t local_face_id) const {
    return trace_face2vol_cell_[static_cast<size_t>(local_face_id)];
  }

  int local_face_in_cell(index_t local_face_id) const {
    return trace_face_local_id_[static_cast<size_t>(local_face_id)];
  }

  // Full incidence to volume mesh (minus/plus ordering)
  //
  // Convention:
  // - Boundary faces: `cell_plus == INVALID_INDEX`
  // - Interior faces: plus-side is the cell with the larger global ID (or local ID if no GIDs)
  std::array<index_t,2> volume_cells(index_t local_face_id) const {
    return trace_face2vol_cells_[static_cast<size_t>(local_face_id)];
  }

  index_t volume_cell_minus(index_t local_face_id) const {
    return trace_face2vol_cells_[static_cast<size_t>(local_face_id)][0];
  }

  index_t volume_cell_plus(index_t local_face_id) const {
    return trace_face2vol_cells_[static_cast<size_t>(local_face_id)][1];
  }

  int local_face_in_cell_minus(index_t local_face_id) const {
    return trace_face_local_ids_[static_cast<size_t>(local_face_id)][0];
  }

  int local_face_in_cell_plus(index_t local_face_id) const {
    return trace_face_local_ids_[static_cast<size_t>(local_face_id)][1];
  }

  bool is_boundary_face(index_t local_face_id) const {
    const auto cells = volume_cells(local_face_id);
    return cells[0] == INVALID_INDEX || cells[1] == INVALID_INDEX;
  }

  // ---- Orientation & normals
  bool has_orientation() const { return !trace_face_orientation_.empty(); }

  perm_code_t face_orientation(index_t local_face_id) const {
    if (!has_orientation()) return -1;
    return trace_face_orientation_[static_cast<size_t>(local_face_id)];
  }

  // Get outward normal at face center (requires volume mesh for geometry)
  std::array<real_t,3> face_normal(index_t local_face_id, const MeshBase& volume_mesh,
                                   Configuration cfg = Configuration::Reference) const {
    const index_t vol_face = trace_face_to_volume_face_.at(static_cast<size_t>(local_face_id));
    return volume_mesh.face_normal(vol_face, cfg);
  }

  // ---- Geometry (requires volume mesh)
  std::array<real_t,3> face_center(index_t local_face_id, const MeshBase& volume_mesh,
                                   Configuration cfg = Configuration::Reference) const {
    auto [vertices_ptr, n_vertices] = face_vertices_span(local_face_id);
    const auto& coords = (cfg == Configuration::Current && volume_mesh.has_current_coords())
                         ? volume_mesh.X_cur() : volume_mesh.X_ref();

    std::array<real_t,3> center = {0, 0, 0};
    for (size_t i = 0; i < n_vertices; ++i) {
      const index_t trace_v = vertices_ptr[i];
      const index_t vol_v = trace_vertex_to_volume_vertex_.at(static_cast<size_t>(trace_v));
      for (int d = 0; d < spatial_dim_; ++d) {
        center[d] += coords[vol_v * spatial_dim_ + d];
      }
    }
    for (int d = 0; d < spatial_dim_; ++d) {
      center[d] /= static_cast<real_t>(n_vertices);
    }
    return center;
  }

  real_t face_area(index_t local_face_id, const MeshBase& volume_mesh,
                   Configuration cfg = Configuration::Reference) const {
    const index_t vol_face = trace_face_to_volume_face_.at(static_cast<size_t>(local_face_id));
    return volume_mesh.face_area(vol_face, cfg);
  }

  // ---- Field attachments (interface fields)
  // Interface-specific fields (e.g., wall shear stress, heat flux, contact pressure)
  FieldHandle attach_field(const std::string& name, FieldScalarType type,
                           size_t components, size_t custom_bytes = 0) {
    auto existing = fields_by_name_.find(name);
    if (existing != fields_by_name_.end()) {
      return existing->second.handle;
    }

    const size_t bpc = (type == FieldScalarType::Custom) ? custom_bytes : bytes_per(type);
    if (bpc == 0) {
      throw std::runtime_error("InterfaceMesh::attach_field: invalid bytes per component for '" + name + "'");
    }

    FieldInfo info;
    info.handle.id = next_field_id_++;
    info.handle.kind = EntityKind::Face;
    info.handle.name = name;
    info.type = type;
    info.components = components;
    info.bytes_per_component = bpc;
    info.data.assign(n_faces() * components * bpc, uint8_t{0});

    fields_by_name_[name] = info;
    fields_by_id_[info.handle.id] = name;
    return info.handle;
  }

  bool has_field(const std::string& name) const {
    return fields_by_name_.find(name) != fields_by_name_.end();
  }

  FieldHandle get_field_handle(const std::string& name) const {
    auto it = fields_by_name_.find(name);
    if (it == fields_by_name_.end()) return {};
    return it->second.handle;
  }

  void* field_data(const FieldHandle& h) {
    auto it_name = fields_by_id_.find(h.id);
    if (it_name == fields_by_id_.end()) return nullptr;
    auto it = fields_by_name_.find(it_name->second);
    if (it == fields_by_name_.end()) return nullptr;
    return it->second.data.data();
  }

  const void* field_data(const FieldHandle& h) const {
    auto it_name = fields_by_id_.find(h.id);
    if (it_name == fields_by_id_.end()) return nullptr;
    auto it = fields_by_name_.find(it_name->second);
    if (it == fields_by_name_.end()) return nullptr;
    return it->second.data.data();
  }

  size_t field_bytes_per_entity(const FieldHandle& h) const {
    auto it_name = fields_by_id_.find(h.id);
    if (it_name == fields_by_id_.end()) return 0;
    auto it = fields_by_name_.find(it_name->second);
    if (it == fields_by_name_.end()) return 0;
    return it->second.components * it->second.bytes_per_component;
  }

  template <typename T>
  T* field_data_as(const FieldHandle& h) {
    return reinterpret_cast<T*>(field_data(h));
  }

  // ---- Label/set system for interface regions
  void set_region_label(index_t local_face_id, label_t label) {
    if (trace_face_region_id_.size() < n_faces()) {
      trace_face_region_id_.resize(n_faces(), -1);
    }
    trace_face_region_id_[static_cast<size_t>(local_face_id)] = label;
  }

  label_t region_label(index_t local_face_id) const {
    if (local_face_id >= static_cast<index_t>(trace_face_region_id_.size())) return -1;
    return trace_face_region_id_[static_cast<size_t>(local_face_id)];
  }

  // ---- Metadata
  const std::string& name() const { return name_; }
  void set_name(const std::string& n) { name_ = n; }

  const std::string& parent_face_set() const { return parent_face_set_; }

private:
  struct FieldInfo {
    FieldHandle handle;
    FieldScalarType type{};
    size_t components = 0;
    size_t bytes_per_component = 0;
    std::vector<uint8_t> data;
  };

  // Metadata
  std::string name_;
  std::string parent_face_set_;
  int spatial_dim_ = 0;

  // Trace topology (local numbering)
  std::vector<gid_t> trace_face_gid_;        // global IDs for trace faces
  std::vector<gid_t> trace_vertex_gid_;      // global IDs for trace vertices
  std::vector<offset_t> trace_face2vertex_offsets_; // CSR offsets
  std::vector<index_t> trace_face2vertex_;   // CSR connectivity (trace-local vertex IDs)

  // Incidence to volume mesh
  std::vector<index_t> trace_face_to_volume_face_; // volume-mesh face id for each trace face
  std::vector<index_t> trace_face2vol_cell_;  // parent volume cell for each trace face
  std::vector<int> trace_face_local_id_;      // local face index within parent cell
  std::vector<std::array<index_t,2>> trace_face2vol_cells_; // (cell_minus, cell_plus)
  std::vector<std::array<int,2>> trace_face_local_ids_;     // (local_face_minus, local_face_plus)
  std::vector<index_t> trace_vertex_to_volume_vertex_; // volume-mesh vertex id for each trace vertex

  // Orientation
  std::vector<perm_code_t> trace_face_orientation_; // permutation code for each face

  // Labels
  std::vector<label_t> trace_face_region_id_;

  // Field management
  uint32_t next_field_id_ = 1;
  std::unordered_map<std::string, FieldInfo> fields_by_name_;
  std::unordered_map<uint32_t, std::string> fields_by_id_;

  // Builder helper
  friend InterfaceMesh build_interface_impl(const MeshBase&, const std::vector<index_t>&, bool);
};

// ====================
// Builder implementation
// ====================
inline InterfaceMesh build_interface_impl(
    const MeshBase& volume_mesh,
    const std::vector<index_t>& face_indices,
    bool compute_orientation)
{
  InterfaceMesh interface;
  interface.spatial_dim_ = volume_mesh.dim();
  interface.trace_face_gid_.reserve(face_indices.size());
  interface.trace_face_to_volume_face_.reserve(face_indices.size());
  interface.trace_face2vol_cell_.reserve(face_indices.size());
  interface.trace_face_local_id_.reserve(face_indices.size());
  interface.trace_face2vol_cells_.reserve(face_indices.size());
  interface.trace_face_local_ids_.reserve(face_indices.size());

  auto compute_local_face_in_cell = [&volume_mesh](index_t cell_id, index_t face_id) -> int {
    if (cell_id < 0) return -1;

    CellTopology::FaceListView view{};
    try {
      view = CellTopology::get_boundary_faces_canonical_view(volume_mesh.cell_shape(cell_id).family);
    } catch (...) {
      return -1;
    }

    auto [face_verts_ptr, n_face_verts] = volume_mesh.face_vertices_span(face_id);
    std::vector<index_t> face_verts(face_verts_ptr, face_verts_ptr + n_face_verts);
    std::sort(face_verts.begin(), face_verts.end());

    auto [cell_verts_ptr, n_cell_verts] = volume_mesh.cell_vertices_span(cell_id);
    (void)n_cell_verts;

    for (int lf = 0; lf < view.face_count; ++lf) {
      const int start = view.offsets[lf];
      const int end = view.offsets[lf + 1];
      std::vector<index_t> cand;
      cand.reserve(static_cast<size_t>(end - start));
      for (int i = start; i < end; ++i) {
        cand.push_back(cell_verts_ptr[view.indices[i]]);
      }
      std::sort(cand.begin(), cand.end());
      if (cand == face_verts) {
        return lf;
      }
    }

    return -1;
  };

  // Collect unique vertices
  std::unordered_set<index_t> vertex_set;
  for (index_t face_id : face_indices) {
    auto [vertices_ptr, n_vertices] = volume_mesh.face_vertices_span(face_id);
    for (size_t i = 0; i < n_vertices; ++i) {
      vertex_set.insert(vertices_ptr[i]);
    }
  }

  std::vector<index_t> trace_vertices(vertex_set.begin(), vertex_set.end());
  std::sort(trace_vertices.begin(), trace_vertices.end());

  // Build vertex mapping
  std::unordered_map<index_t, index_t> vol_to_trace_vertex;
  for (size_t i = 0; i < trace_vertices.size(); ++i) {
    vol_to_trace_vertex[trace_vertices[i]] = static_cast<index_t>(i);
    interface.trace_vertex_to_volume_vertex_.push_back(trace_vertices[i]);
    // Set vertex GID (if available)
    const auto& vertex_gids_alias = volume_mesh.vertex_gids();
    if (!vertex_gids_alias.empty() && trace_vertices[i] < static_cast<index_t>(vertex_gids_alias.size())) {
      interface.trace_vertex_gid_.push_back(vertex_gids_alias[trace_vertices[i]]);
    } else {
      interface.trace_vertex_gid_.push_back(static_cast<gid_t>(trace_vertices[i]));
    }
  }

  // Build face topology
  interface.trace_face2vertex_offsets_.push_back(0);
  for (index_t face_id : face_indices) {
    interface.trace_face_to_volume_face_.push_back(face_id);

    // Set face GID
    const auto& face_gids = volume_mesh.face_gids();
    if (!face_gids.empty() && face_id < static_cast<index_t>(face_gids.size())) {
      interface.trace_face_gid_.push_back(face_gids[face_id]);
    } else {
      interface.trace_face_gid_.push_back(static_cast<gid_t>(face_id));
    }

    // Get face vertices and map to trace-local IDs
    auto [vertices_ptr, n_vertices] = volume_mesh.face_vertices_span(face_id);
    for (size_t i = 0; i < n_vertices; ++i) {
      interface.trace_face2vertex_.push_back(vol_to_trace_vertex[vertices_ptr[i]]);
    }
    interface.trace_face2vertex_offsets_.push_back(static_cast<offset_t>(interface.trace_face2vertex_.size()));

    // Store incidence to volume mesh
    const auto& face_cells = volume_mesh.face_cells(face_id);
    index_t cell_minus = face_cells[0];
    index_t cell_plus = face_cells[1];

    // Boundary faces: ensure minus-side is the valid cell.
    if (cell_minus == INVALID_INDEX && cell_plus != INVALID_INDEX) {
      cell_minus = cell_plus;
      cell_plus = INVALID_INDEX;
    }

    // Interior faces: sort by global ID when available (DG-style plus/minus convention).
    if (cell_minus != INVALID_INDEX && cell_plus != INVALID_INDEX) {
      const auto& cell_gids = volume_mesh.cell_gids();
      const gid_t gid_minus = (!cell_gids.empty() && cell_minus < static_cast<index_t>(cell_gids.size()))
                                ? cell_gids[static_cast<size_t>(cell_minus)]
                                : static_cast<gid_t>(cell_minus);
      const gid_t gid_plus = (!cell_gids.empty() && cell_plus < static_cast<index_t>(cell_gids.size()))
                               ? cell_gids[static_cast<size_t>(cell_plus)]
                               : static_cast<gid_t>(cell_plus);
      if (gid_minus > gid_plus) {
        std::swap(cell_minus, cell_plus);
      }
    }

    interface.trace_face2vol_cells_.push_back({cell_minus, cell_plus});
    interface.trace_face_local_ids_.push_back(
        {compute_local_face_in_cell(cell_minus, face_id),
         compute_local_face_in_cell(cell_plus, face_id)});

    // Backward-compatible "primary" parent accessors use the minus side.
    interface.trace_face2vol_cell_.push_back(cell_minus);
    interface.trace_face_local_id_.push_back(interface.trace_face_local_ids_.back()[0]);
  }

  // Compute orientation if requested
  if (compute_orientation) {
    interface.trace_face_orientation_.assign(face_indices.size(), -1);

    OrientationManager orient(volume_mesh);
    orient.build();

    for (size_t i = 0; i < face_indices.size(); ++i) {
      const index_t cell = interface.trace_face2vol_cell_[i];
      const int lf = interface.trace_face_local_id_[i];
      if (cell >= 0 && lf >= 0) {
        interface.trace_face_orientation_[i] = orient.face_orientation(cell, lf);
      }
    }
  }

  return interface;
}

inline InterfaceMesh InterfaceMesh::build_from_face_set(
    const MeshBase& volume_mesh,
    const std::string& face_set_name,
    bool compute_orientation)
{
  if (!volume_mesh.has_set(EntityKind::Face, face_set_name)) {
    throw std::runtime_error("InterfaceMesh: face set '" + face_set_name + "' not found");
  }

  const auto& face_indices = volume_mesh.get_set(EntityKind::Face, face_set_name);
  auto interface = build_interface_impl(volume_mesh, face_indices, compute_orientation);
  interface.name_ = face_set_name + "_interface";
  interface.parent_face_set_ = face_set_name;
  return interface;
}

inline InterfaceMesh InterfaceMesh::build_from_boundary_label(
    const MeshBase& volume_mesh,
    label_t boundary_label,
    bool compute_orientation)
{
  auto face_indices = volume_mesh.faces_with_label(boundary_label);
  auto interface = build_interface_impl(volume_mesh, face_indices, compute_orientation);
  interface.name_ = "boundary_" + std::to_string(boundary_label) + "_interface";
  return interface;
}

inline InterfaceMesh InterfaceMesh::build_from_region_boundary(
    const MeshBase& volume_mesh,
    label_t region_label,
    bool compute_orientation)
{
  // Extract all boundary faces of cells with given region label
  auto cells = volume_mesh.cells_with_label(region_label);
  std::vector<index_t> face_indices;

  // For each cell, find its boundary faces
  // (simplified; production would use cell2face topology)
  const auto& boundary_faces = volume_mesh.boundary_faces();
  for (index_t face_id : boundary_faces) {
    const auto& face_cells = volume_mesh.face_cells(face_id);
    if (face_cells[0] >= 0 && volume_mesh.region_label(face_cells[0]) == region_label) {
      face_indices.push_back(face_id);
    }
  }

  auto interface = build_interface_impl(volume_mesh, face_indices, compute_orientation);
  interface.name_ = "region_" + std::to_string(region_label) + "_boundary";
  return interface;
}

} // namespace svmp

#endif // SVMP_INTERFACE_MESH_H
