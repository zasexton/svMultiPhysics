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
#include <memory>
#include <string>
#include <vector>

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
  index_t volume_cell(index_t local_face_id) const {
    return trace_face2vol_cell_[static_cast<size_t>(local_face_id)];
  }

  int local_face_in_cell(index_t local_face_id) const {
    return trace_face_local_id_[static_cast<size_t>(local_face_id)];
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
    index_t vol_cell = volume_cell(local_face_id);
    int local_f = local_face_in_cell(local_face_id);
    // Compute normal from volume mesh geometry
    // (simplified; production would use face vertices and orientation)
    return volume_mesh.face_normal(volume_mesh.boundary_faces()[local_f], cfg);
  }

  // ---- Geometry (requires volume mesh)
  std::array<real_t,3> face_center(index_t local_face_id, const MeshBase& volume_mesh,
                                   Configuration cfg = Configuration::Reference) const {
    auto [vertices_ptr, n_vertices] = face_vertices_span(local_face_id);
    const auto& coords = (cfg == Configuration::Current && volume_mesh.has_current_coords())
                         ? volume_mesh.X_cur() : volume_mesh.X_ref();

    std::array<real_t,3> center = {0, 0, 0};
    for (size_t i = 0; i < n_vertices; ++i) {
      index_t vertex_id = vertices_ptr[i];
      // Need to map trace vertex to volume vertex via GID
      // (simplified; production would maintain trace_vertex_to_volume_vertex map)
      for (int d = 0; d < spatial_dim_; ++d) {
        center[d] += coords[vertex_id * spatial_dim_ + d];
      }
    }
    for (int d = 0; d < spatial_dim_; ++d) {
      center[d] /= static_cast<real_t>(n_vertices);
    }
    return center;
  }

  real_t face_area(index_t local_face_id, const MeshBase& volume_mesh,
                   Configuration cfg = Configuration::Reference) const {
    index_t vol_cell = volume_cell(local_face_id);
    int local_f = local_face_in_cell(local_face_id);
    // Use volume mesh face area computation
    // (simplified; production would access via face GID)
    const auto& boundary_faces = volume_mesh.boundary_faces();
    if (local_f >= 0 && local_f < static_cast<int>(boundary_faces.size())) {
      return volume_mesh.face_area(boundary_faces[local_f], cfg);
    }
    return 0.0;
  }

  // ---- Field attachments (interface fields)
  // Interface-specific fields (e.g., wall shear stress, heat flux, contact pressure)
  MeshBase::FieldHandle attach_field(const std::string& name, FieldScalarType type,
                                     size_t components, size_t custom_bytes = 0) {
    // Store field data locally (simplified version)
    // Production would use a full field attachment system
    MeshBase::FieldHandle h;
    h.id = next_field_id_++;
    h.kind = EntityKind::Face;
    h.name = name;
    // TODO: allocate storage
    return h;
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
  std::vector<index_t> trace_face2vol_cell_;  // parent volume cell for each trace face
  std::vector<int> trace_face_local_id_;      // local face index within parent cell

  // Orientation
  std::vector<perm_code_t> trace_face_orientation_; // permutation code for each face

  // Labels
  std::vector<label_t> trace_face_region_id_;

  // Field management
  uint32_t next_field_id_ = 1;

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
  interface.trace_face2vol_cell_.reserve(face_indices.size());
  interface.trace_face_local_id_.reserve(face_indices.size());

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
    interface.trace_face2vol_cell_.push_back(face_cells[0]); // inner cell
    interface.trace_face_local_id_.push_back(-1); // TODO: compute local face index
  }

  // Compute orientation if requested
  if (compute_orientation) {
    interface.trace_face_orientation_.resize(face_indices.size(), 0);
    // TODO: implement orientation computation using OrientationManager
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
