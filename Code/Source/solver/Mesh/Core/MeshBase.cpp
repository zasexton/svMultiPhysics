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

#include "MeshBase.h"
#include "../Geometry/MeshGeometry.h"
#include "../Geometry/MeshQuality.h"
#include "../Topology/MeshTopology.h"
#include "../Topology/CellTopology.h"
#include "../Search/MeshSearch.h"
#include "../Validation/MeshValidation.h"
#include "../Labels/MeshLabels.h"
#include "../Fields/MeshFields.h"
#include "../Boundary/BoundaryKey.h"

#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <cstring>

namespace svmp {

// ==========================================
// Constructors and Lifecycle
// ==========================================

MeshBase::MeshBase()
    : spatial_dim_(0),
      active_config_(Configuration::Reference),
      next_field_id_(1) {
}

MeshBase::MeshBase(int spatial_dim)
    : spatial_dim_(spatial_dim),
      active_config_(Configuration::Reference),
      next_field_id_(1) {
    if (spatial_dim < 0 || spatial_dim > 3) {
        throw std::invalid_argument("MeshBase: spatial dimension must be 0, 1, 2, or 3");
    }
}

// ==========================================
// Builder Methods
// ==========================================

void MeshBase::clear() {
    // Clear coordinates
    X_ref_.clear();
    X_cur_.clear();

    // Clear topology
    cell_shape_.clear();
    cell2vertex_offsets_.clear();
    cell2vertex_.clear();

    face_shape_.clear();
    face2vertex_offsets_.clear();
    face2vertex_.clear();
    face2cell_.clear();

    edge2vertex_.clear();

    // Clear IDs
    vertex_gid_.clear();
    cell_gid_.clear();
    face_gid_.clear();
    edge_gid_.clear();

    // Clear ownership
    vertex_owner_.clear();
    cell_owner_.clear();
    face_owner_.clear();
    edge_owner_.clear();

    // Clear labels
    cell_region_id_.clear();
    face_boundary_id_.clear();
    for (int i = 0; i < 4; ++i) {
        entity_sets_[i].clear();
    }
    label_from_name_.clear();
    name_from_label_.clear();

    // Clear fields
    for (int i = 0; i < 4; ++i) {
        attachments_[i].by_name.clear();
    }
    field_index_.clear();
    field_descriptors_.clear();
    next_field_id_ = 1;

    // Clear adjacency caches
    invalidate_caches();

    // Clear global to local maps
    global2local_cell_.clear();
    global2local_vertex_.clear();
    global2local_face_.clear();
    global2local_edge_.clear();

    // Clear search structure
    clear_search_structure();

    // Reset dimension
    spatial_dim_ = 0;
    active_config_ = Configuration::Reference;

    // Notify observers
    event_bus_.notify(MeshEvent::TopologyChanged);
    event_bus_.notify(MeshEvent::GeometryChanged);
}

void MeshBase::reserve(index_t n_vertices, index_t n_cells, index_t n_faces) {
    // Reserve coordinate storage
    X_ref_.reserve(n_vertices * spatial_dim_);

    // Reserve cell storage
    cell_shape_.reserve(n_cells);
    cell2vertex_offsets_.reserve(n_cells + 1);
    // Estimate connectivity (8 vertices per cell average)
    cell2vertex_.reserve(n_cells * 8);

    // Reserve face storage if specified
    if (n_faces > 0) {
        face_shape_.reserve(n_faces);
        face2vertex_offsets_.reserve(n_faces + 1);
        face2vertex_.reserve(n_faces * 4);
        face2cell_.reserve(n_faces);
    }

    // Reserve labels
    cell_region_id_.reserve(n_cells);

    // Reserve global IDs
    vertex_gid_.reserve(n_vertices);
    cell_gid_.reserve(n_cells);
}

//
// Connectivity storage note (cell2vertex_offsets / cell2vertex):
// We use a CSR-style layout to store variable-length vertex lists per cell.
// - The array `cell2vertex_offsets` has length n_cells + 1 and starts at 0.
// - For cell c, its vertices are in `cell2vertex[start:end]` where
//     start = cell2vertex_offsets[c]
//     end   = cell2vertex_offsets[c+1]
// Example:
//   offsets = [0, 3, 7, 10]
//   cell2vertex = [0,1,2,  0,3,2,4,  5,6,7]
//   => cell 0 -> [0,1,2]
//      cell 1 -> [0,3,2,4]
//      cell 2 -> [5,6,7]
// This compact scheme supports mixed element types without padding and scales
// to very large meshes (offset_t is 64-bit for >2B entries).
void MeshBase::build_from_arrays(
    int spatial_dim,
    const std::vector<real_t>& X_ref,
    const std::vector<offset_t>& cell2vertex_offsets,
    const std::vector<index_t>& cell2vertex,
    const std::vector<CellShape>& cell_shape) {

    // Validate inputs
    if (spatial_dim < 1 || spatial_dim > 3) {
        throw std::invalid_argument("build_from_arrays: invalid spatial dimension");
    }

    if (cell2vertex_offsets.empty() || cell2vertex_offsets[0] != 0) {
        throw std::invalid_argument("build_from_arrays: invalid offsets array");
    }

    if (cell_shape.size() != cell2vertex_offsets.size() - 1) {
        throw std::invalid_argument("build_from_arrays: shape and offset sizes don't match");
    }

    size_t n_vertices = X_ref.size() / spatial_dim;
    if (X_ref.size() != n_vertices * spatial_dim) {
        throw std::invalid_argument("build_from_arrays: coordinate array size mismatch");
    }

    // Clear existing data
    clear();

    // Set dimension
    spatial_dim_ = spatial_dim;

    // Copy data
    X_ref_ = X_ref;
    cell2vertex_offsets_ = cell2vertex_offsets;
    cell2vertex_ = cell2vertex;
    cell_shape_ = cell_shape;

    // Initialize region labels to 0
    cell_region_id_.resize(cell_shape_.size(), 0);

    // Build global IDs (default to local indices)
    vertex_gid_.resize(n_vertices);
    std::iota(vertex_gid_.begin(), vertex_gid_.end(), 0);

    cell_gid_.resize(cell_shape_.size());
    std::iota(cell_gid_.begin(), cell_gid_.end(), 0);

    // Build global to local maps
    for (index_t i = 0; i < static_cast<index_t>(n_vertices); ++i) {
        global2local_vertex_[vertex_gid_[i]] = i;
    }
    for (index_t i = 0; i < static_cast<index_t>(cell_shape_.size()); ++i) {
        global2local_cell_[cell_gid_[i]] = i;
    }

    // Notify observers
    event_bus_.notify(MeshEvent::TopologyChanged);
    event_bus_.notify(MeshEvent::GeometryChanged);
}

void MeshBase::set_faces_from_arrays(
    const std::vector<CellShape>& face_shape,
    const std::vector<offset_t>& face2vertex_offsets,
    const std::vector<index_t>& face2vertex,
    const std::vector<std::array<index_t,2>>& face2cell) {

    // Validate inputs
    if (face_shape.size() != face2vertex_offsets.size() - 1) {
        throw std::invalid_argument("set_faces_from_arrays: shape and offset sizes don't match");
    }

    if (face_shape.size() != face2cell.size()) {
        throw std::invalid_argument("set_faces_from_arrays: shape and face2cell sizes don't match");
    }

    // Copy face data
    face_shape_ = face_shape;
    face2vertex_offsets_ = face2vertex_offsets;
    face2vertex_ = face2vertex;
    face2cell_ = face2cell;

    // Initialize boundary labels to INVALID_LABEL (unlabeled)
    face_boundary_id_.resize(face_shape.size(), INVALID_LABEL);

    // Build face global IDs
    face_gid_.resize(face_shape.size());
    std::iota(face_gid_.begin(), face_gid_.end(), 0);

    for (index_t i = 0; i < static_cast<index_t>(face_shape.size()); ++i) {
        global2local_face_[face_gid_[i]] = i;
    }

    // Topology changed -> invalidate caches and notify observers
    invalidate_caches();
    event_bus_.notify(MeshEvent::TopologyChanged);
}

void MeshBase::set_edges_from_arrays(const std::vector<std::array<index_t,2>>& edge2vertex) {
    edge2vertex_ = edge2vertex;

    // Build edge global IDs
    edge_gid_.resize(edge2vertex.size());
    std::iota(edge_gid_.begin(), edge_gid_.end(), 0);

    // Build edge global-to-local map
    global2local_edge_.clear();
    for (index_t i = 0; i < static_cast<index_t>(edge_gid_.size()); ++i) {
        global2local_edge_[edge_gid_[i]] = i;
    }

    // Topology changed -> invalidate caches and notify observers
    invalidate_caches();
    event_bus_.notify(MeshEvent::TopologyChanged);
}

void MeshBase::finalize() {
    // Build face connectivity if not provided. Faces are (n-1)-entities:
    // - In 3D meshes: polygonal faces
    // - In 2D meshes: edges (lines)
    if (face_shape_.empty() && !cell_shape_.empty() && spatial_dim_ >= 2) {
        struct FaceAcc {
            std::vector<index_t> oriented_vertices;
            std::vector<index_t> incident_cells;
            CellFamily family = CellFamily::Polygon;
            int num_corners = 0;
            bool family_set = false;
        };

        std::unordered_map<BoundaryKey, FaceAcc, BoundaryKey::Hash> face_map;

        // Enumerate (n-1)-entities from each cell using CellTopology definitions
        for (index_t c = 0; c < static_cast<index_t>(cell_shape_.size()); ++c) {
            auto [vertices_ptr, n_vertices] = cell_vertices_span(c);
            const auto& cshape = cell_shape(c);

            // Handle standard families via CellTopology; handle 2D polygons explicitly.
            if ((cshape.family == CellFamily::Triangle) ||
                (cshape.family == CellFamily::Quad)     ||
                (cshape.family == CellFamily::Tetra)    ||
                (cshape.family == CellFamily::Hex)      ||
                (cshape.family == CellFamily::Wedge)    ||
                (cshape.family == CellFamily::Pyramid)) {
                auto face_defs = CellTopology::get_boundary_faces(cshape.family);
                auto oriented_defs = CellTopology::get_oriented_boundary_faces(cshape.family);

                for (size_t i = 0; i < face_defs.size(); ++i) {
                    // Canonical key (sorted vertices)
                    std::vector<index_t> face_vertices;
                    face_vertices.reserve(face_defs[i].size());
                    for (index_t li : face_defs[i]) face_vertices.push_back(vertices_ptr[li]);
                    BoundaryKey key(face_vertices);

                    // Oriented vertices (preserve ordering)
                    std::vector<index_t> orn;
                    orn.reserve(oriented_defs[i].size());
                    for (index_t li : oriented_defs[i]) orn.push_back(vertices_ptr[li]);

                    auto& acc = face_map[key];
                    if (acc.oriented_vertices.empty()) acc.oriented_vertices = std::move(orn);
                    acc.incident_cells.push_back(c);

                    // Derive face family from mesh dimension and cell topology
                    if (!acc.family_set) {
                        if (spatial_dim_ == 2) {
                            acc.family = CellFamily::Line;  // faces of 2D cells are edges
                        } else {
                            const size_t fsize = face_defs[i].size();
                            acc.family = (fsize == 3) ? CellFamily::Triangle
                                        : (fsize == 4) ? CellFamily::Quad
                                                       : CellFamily::Polygon;
                        }
                        acc.num_corners = static_cast<int>(face_defs[i].size());
                        acc.family_set = true;
                    }
                }
            } else if (cshape.family == CellFamily::Polygon && spatial_dim_ == 2) {
                // Build edges as codim-1 entities for a 2D polygon by connecting consecutive vertices
                const int nc = (cshape.num_corners > 0) ? cshape.num_corners : static_cast<int>(n_vertices);
                if (nc >= 2) {
                    for (int i = 0; i < nc; ++i) {
                        index_t a = vertices_ptr[i];
                        index_t b = vertices_ptr[(i + 1) % nc];
                        std::vector<index_t> face_vertices = {a, b};
                        BoundaryKey key(face_vertices);

                        std::vector<index_t> orn = {a, b};
                        auto& acc = face_map[key];
                        if (acc.oriented_vertices.empty()) acc.oriented_vertices = std::move(orn);
                        acc.incident_cells.push_back(c);
                        if (!acc.family_set) {
                            acc.family = CellFamily::Line;
                            acc.num_corners = 2;
                            acc.family_set = true;
                        }
                    }
                }
            } else {
                // Unsupported for codim-1 extraction (e.g., 3D polyhedra without explicit face lists)
                // Skip gracefully.
                continue;
            }
        }

        // Finalize arrays
        std::vector<CellShape> new_face_shape;
        std::vector<offset_t>  new_face2vertex_offsets;
        std::vector<index_t>   new_face2vertex;
        std::vector<std::array<index_t,2>> new_face2cell;

        new_face2vertex_offsets.reserve(face_map.size() + 1);
        new_face2vertex_offsets.push_back(0);
        new_face_shape.reserve(face_map.size());
        new_face2cell.reserve(face_map.size());

        for (const auto& kv : face_map) {
            const auto& acc = kv.second;

            // Connectivity
            for (index_t v : acc.oriented_vertices) new_face2vertex.push_back(v);
            new_face2vertex_offsets.push_back(static_cast<offset_t>(new_face2vertex.size()));

            // Shape
            CellShape fshape;
            fshape.family = acc.family;
            fshape.num_corners = acc.num_corners;
            fshape.order = 1;
            fshape.is_mixed_order = false;
            new_face_shape.push_back(fshape);

            // Adjacent cells (up to 2)
            std::array<index_t,2> adj = {{INVALID_INDEX, INVALID_INDEX}};
            if (!acc.incident_cells.empty()) adj[0] = acc.incident_cells[0];
            if (acc.incident_cells.size() > 1) adj[1] = acc.incident_cells[1];
            new_face2cell.push_back(adj);
        }

        set_faces_from_arrays(new_face_shape, new_face2vertex_offsets, new_face2vertex, new_face2cell);
    }

    // Build edge connectivity for 2D/3D meshes if not provided
    if (edge2vertex_.empty() && spatial_dim_ >= 2 && !cell_shape_.empty()) {
        auto edges = MeshTopology::extract_edges(*this);
        set_edges_from_arrays(edges);
    }

    // Validate basic mesh consistency
    validate_basic();

    // No event: finalize() is a check; mutating calls already notified

    // Populate num_faces_hint for polyhedra when face connectivity is available
    if (!face2cell_.empty()) {
        std::vector<int> face_count_per_cell(cell_shape_.size(), 0);
        for (const auto& fc : face2cell_) {
            if (fc[0] != INVALID_INDEX) face_count_per_cell[static_cast<size_t>(fc[0])]++;
            if (fc[1] != INVALID_INDEX) face_count_per_cell[static_cast<size_t>(fc[1])]++;
        }
        for (size_t c = 0; c < cell_shape_.size(); ++c) {
            if (cell_shape_[c].family == CellFamily::Polyhedron) {
                cell_shape_[c].num_faces_hint = face_count_per_cell[c];
            }
        }
    }
}

// ==========================================
// Coordinate Management
// ==========================================

void MeshBase::set_current_coords(const std::vector<real_t>& Xcur) {
    if (Xcur.size() != X_ref_.size()) {
        throw std::invalid_argument("set_current_coords: size mismatch with reference coordinates");
    }
    X_cur_ = Xcur;
    event_bus_.notify(MeshEvent::GeometryChanged);
}

void MeshBase::clear_current_coords() {
    X_cur_.clear();
    event_bus_.notify(MeshEvent::GeometryChanged);
}

// ==========================================
// Topology Access
// ==========================================

std::pair<const index_t*, size_t> MeshBase::cell_vertices_span(index_t c) const {
    if (c < 0 || c >= static_cast<index_t>(cell_shape_.size())) {
        throw std::out_of_range("cell_vertices_span: invalid cell index");
    }

    offset_t start = cell2vertex_offsets_[c];
    offset_t end = cell2vertex_offsets_[c + 1];
    size_t count = static_cast<size_t>(end - start);

    return {&cell2vertex_[start], count};
}

std::pair<const index_t*, size_t> MeshBase::face_vertices_span(index_t f) const {
    if (f < 0 || f >= static_cast<index_t>(face_shape_.size())) {
        throw std::out_of_range("face_vertices_span: invalid face index");
    }

    offset_t start = face2vertex_offsets_[f];
    offset_t end = face2vertex_offsets_[f + 1];
    size_t count = static_cast<size_t>(end - start);

    return {&face2vertex_[start], count};
}

// ==========================================
// Global to Local ID Mapping
// ==========================================

index_t MeshBase::global_to_local_cell(gid_t gid) const {
    auto it = global2local_cell_.find(gid);
    if (it != global2local_cell_.end()) {
        return it->second;
    }
    return INVALID_INDEX;
}

index_t MeshBase::global_to_local_vertex(gid_t gid) const {
    auto it = global2local_vertex_.find(gid);
    if (it != global2local_vertex_.end()) {
        return it->second;
    }
    return INVALID_INDEX;
}

index_t MeshBase::global_to_local_face(gid_t gid) const {
    auto it = global2local_face_.find(gid);
    if (it != global2local_face_.end()) {
        return it->second;
    }
    return INVALID_INDEX;
}

index_t MeshBase::global_to_local_edge(gid_t gid) const {
    auto it = global2local_edge_.find(gid);
    if (it != global2local_edge_.end()) {
        return it->second;
    }
    return INVALID_INDEX;
}

// ==========================================
// Labels and Sets
// ==========================================

void MeshBase::set_region_label(index_t cell, label_t label) {
    if (cell < 0 || cell >= static_cast<index_t>(cell_shape_.size())) {
        throw std::out_of_range("set_region_label: invalid cell index");
    }
    if (cell_region_id_.size() != cell_shape_.size()) {
        cell_region_id_.resize(cell_shape_.size(), 0);
    }
    cell_region_id_[cell] = label;
    event_bus_.notify(MeshEvent::LabelsChanged);
}

label_t MeshBase::region_label(index_t cell) const {
    if (cell < 0 || cell >= static_cast<index_t>(cell_region_id_.size())) {
        return INVALID_LABEL;
    }
    return cell_region_id_[cell];
}

std::vector<index_t> MeshBase::cells_with_label(label_t label) const {
    std::vector<index_t> cells;
    for (index_t i = 0; i < static_cast<index_t>(cell_region_id_.size()); ++i) {
        if (cell_region_id_[i] == label) {
            cells.push_back(i);
        }
    }
    return cells;
}

void MeshBase::set_boundary_label(index_t face, label_t label) {
    if (face < 0 || face >= static_cast<index_t>(face_shape_.size())) {
        throw std::out_of_range("set_boundary_label: invalid face index");
    }
    if (face_boundary_id_.size() != face_shape_.size()) {
        face_boundary_id_.resize(face_shape_.size(), INVALID_LABEL);
    }
    face_boundary_id_[face] = label;
    event_bus_.notify(MeshEvent::LabelsChanged);
}

label_t MeshBase::boundary_label(index_t face) const {
    if (face < 0 || face >= static_cast<index_t>(face_boundary_id_.size())) {
        return INVALID_LABEL;
    }
    return face_boundary_id_[face];
}

std::vector<index_t> MeshBase::faces_with_label(label_t label) const {
    std::vector<index_t> faces;
    for (index_t i = 0; i < static_cast<index_t>(face_boundary_id_.size()); ++i) {
        if (face_boundary_id_[i] == label) {
            faces.push_back(i);
        }
    }
    return faces;
}

void MeshBase::add_to_set(EntityKind kind, const std::string& name, index_t id) {
    int kind_idx = static_cast<int>(kind);
    entity_sets_[kind_idx][name].push_back(id);
    event_bus_.notify(MeshEvent::LabelsChanged);
}

const std::vector<index_t>& MeshBase::get_set(EntityKind kind, const std::string& name) const {
    static const std::vector<index_t> empty;
    int kind_idx = static_cast<int>(kind);
    auto it = entity_sets_[kind_idx].find(name);
    if (it != entity_sets_[kind_idx].end()) {
        return it->second;
    }
    return empty;
}

bool MeshBase::has_set(EntityKind kind, const std::string& name) const {
    int kind_idx = static_cast<int>(kind);
    return entity_sets_[kind_idx].find(name) != entity_sets_[kind_idx].end();
}

void MeshBase::register_label(const std::string& name, label_t label) {
    label_from_name_[name] = label;
    if (label >= static_cast<label_t>(name_from_label_.size())) {
        name_from_label_.resize(label + 1);
    }
    name_from_label_[label] = name;
    event_bus_.notify(MeshEvent::LabelsChanged);
}

std::string MeshBase::label_name(label_t label) const {
    if (label >= 0 && label < static_cast<label_t>(name_from_label_.size())) {
        return name_from_label_[label];
    }
    return "";
}

label_t MeshBase::label_from_name(const std::string& name) const {
    auto it = label_from_name_.find(name);
    if (it != label_from_name_.end()) {
        return it->second;
    }
    return INVALID_LABEL;
}

// ==========================================
// Field Attachment System
// ==========================================

FieldHandle MeshBase::attach_field(EntityKind kind, const std::string& name,
                                  FieldScalarType type, size_t components,
                                  size_t custom_bytes_per_component) {
    int kind_idx = static_cast<int>(kind);

    // Check if field already exists
    if (attachments_[kind_idx].by_name.find(name) != attachments_[kind_idx].by_name.end()) {
        throw std::runtime_error("attach_field: field '" + name + "' already exists");
    }

    // Determine bytes per component
    size_t bytes_per_comp = (type == FieldScalarType::Custom)
        ? custom_bytes_per_component
        : bytes_per(type);

    if (bytes_per_comp == 0) {
        throw std::invalid_argument("attach_field: invalid bytes per component");
    }

    // Get entity count
    size_t n_entities = entity_count(kind);

    // Create field info
    FieldInfo info;
    info.type = type;
    info.components = components;
    info.bytes_per_component = bytes_per_comp;
    info.data.resize(n_entities * components * bytes_per_comp, 0);

    // Store field
    attachments_[kind_idx].by_name[name] = std::move(info);

    // Create handle
    FieldHandle handle;
    handle.id = next_field_id_++;
    handle.kind = kind;
    handle.name = name;

    // Register in index
    field_index_[handle.id] = {kind, name};

    // Notify
    event_bus_.notify(MeshEvent::FieldsChanged);

    return handle;
}

FieldHandle MeshBase::attach_field_with_descriptor(EntityKind kind, const std::string& name,
                                                  FieldScalarType type, const FieldDescriptor& descriptor) {
    FieldHandle handle = attach_field(kind, name, type, descriptor.components);
    field_descriptors_[handle.id] = descriptor;
    return handle;
}

bool MeshBase::has_field(EntityKind kind, const std::string& name) const {
    int kind_idx = static_cast<int>(kind);
    return attachments_[kind_idx].by_name.find(name) != attachments_[kind_idx].by_name.end();
}

void MeshBase::remove_field(const FieldHandle& h) {
    auto it = field_index_.find(h.id);
    if (it == field_index_.end()) {
        return;  // Field doesn't exist
    }

    EntityKind kind = it->second.first;
    const std::string& name = it->second.second;
    int kind_idx = static_cast<int>(kind);

    // Remove from attachments
    attachments_[kind_idx].by_name.erase(name);

    // Remove from indices
    field_index_.erase(h.id);
    field_descriptors_.erase(h.id);

    // Notify
    event_bus_.notify(MeshEvent::FieldsChanged);
}

void* MeshBase::field_data(const FieldHandle& h) {
    auto it = field_index_.find(h.id);
    if (it == field_index_.end()) {
        return nullptr;
    }

    EntityKind kind = it->second.first;
    const std::string& name = it->second.second;
    int kind_idx = static_cast<int>(kind);

    auto field_it = attachments_[kind_idx].by_name.find(name);
    if (field_it == attachments_[kind_idx].by_name.end()) {
        return nullptr;
    }

    return field_it->second.data.data();
}

const void* MeshBase::field_data(const FieldHandle& h) const {
    return const_cast<MeshBase*>(this)->field_data(h);
}

size_t MeshBase::field_components(const FieldHandle& h) const {
    auto it = field_index_.find(h.id);
    if (it == field_index_.end()) {
        return 0;
    }

    EntityKind kind = it->second.first;
    const std::string& name = it->second.second;
    int kind_idx = static_cast<int>(kind);

    auto field_it = attachments_[kind_idx].by_name.find(name);
    if (field_it == attachments_[kind_idx].by_name.end()) {
        return 0;
    }

    return field_it->second.components;
}

FieldScalarType MeshBase::field_type(const FieldHandle& h) const {
    auto it = field_index_.find(h.id);
    if (it == field_index_.end()) {
        return FieldScalarType::Custom;
    }

    EntityKind kind = it->second.first;
    const std::string& name = it->second.second;
    int kind_idx = static_cast<int>(kind);

    auto field_it = attachments_[kind_idx].by_name.find(name);
    if (field_it == attachments_[kind_idx].by_name.end()) {
        return FieldScalarType::Custom;
    }

    return field_it->second.type;
}

size_t MeshBase::field_entity_count(const FieldHandle& h) const {
    auto it = field_index_.find(h.id);
    if (it == field_index_.end()) {
        return 0;
    }

    return entity_count(it->second.first);
}

size_t MeshBase::field_bytes_per_entity(const FieldHandle& h) const {
    auto it = field_index_.find(h.id);
    if (it == field_index_.end()) {
        return 0;
    }

    EntityKind kind = it->second.first;
    const std::string& name = it->second.second;
    int kind_idx = static_cast<int>(kind);

    auto field_it = attachments_[kind_idx].by_name.find(name);
    if (field_it == attachments_[kind_idx].by_name.end()) {
        return 0;
    }

    return field_it->second.components * field_it->second.bytes_per_component;
}

std::vector<std::string> MeshBase::field_names(EntityKind kind) const {
    std::vector<std::string> names;
    int kind_idx = static_cast<int>(kind);

    for (const auto& [name, info] : attachments_[kind_idx].by_name) {
        names.push_back(name);
    }

    return names;
}

void* MeshBase::field_data_by_name(EntityKind kind, const std::string& name) {
    int kind_idx = static_cast<int>(kind);

    auto it = attachments_[kind_idx].by_name.find(name);
    if (it == attachments_[kind_idx].by_name.end()) {
        return nullptr;
    }

    return it->second.data.data();
}

const void* MeshBase::field_data_by_name(EntityKind kind, const std::string& name) const {
    return const_cast<MeshBase*>(this)->field_data_by_name(kind, name);
}

size_t MeshBase::field_components_by_name(EntityKind kind, const std::string& name) const {
    int kind_idx = static_cast<int>(kind);

    auto it = attachments_[kind_idx].by_name.find(name);
    if (it == attachments_[kind_idx].by_name.end()) {
        return 0;
    }

    return it->second.components;
}

FieldScalarType MeshBase::field_type_by_name(EntityKind kind, const std::string& name) const {
    int kind_idx = static_cast<int>(kind);

    auto it = attachments_[kind_idx].by_name.find(name);
    if (it == attachments_[kind_idx].by_name.end()) {
        return FieldScalarType::Custom;
    }

    return it->second.type;
}

size_t MeshBase::field_bytes_per_component_by_name(EntityKind kind, const std::string& name) const {
    int kind_idx = static_cast<int>(kind);

    auto it = attachments_[kind_idx].by_name.find(name);
    if (it == attachments_[kind_idx].by_name.end()) {
        return 0;
    }

    return it->second.bytes_per_component;
}

// ==========================================
// Geometry Operations (Delegated)
// ==========================================

std::array<real_t,3> MeshBase::cell_center(index_t c, Configuration cfg) const {
    return MeshGeometry::cell_center(*this, c, cfg);
}

std::array<real_t,3> MeshBase::face_center(index_t f, Configuration cfg) const {
    return MeshGeometry::face_center(*this, f, cfg);
}

std::array<real_t,3> MeshBase::face_normal(index_t f, Configuration cfg) const {
    return MeshGeometry::face_normal(*this, f, cfg);
}

real_t MeshBase::face_area(index_t f, Configuration cfg) const {
    return MeshGeometry::face_area(*this, f, cfg);
}

real_t MeshBase::cell_measure(index_t c, Configuration cfg) const {
    return MeshGeometry::cell_measure(*this, c, cfg);
}

BoundingBox MeshBase::bounding_box(Configuration cfg) const {
    return MeshGeometry::bounding_box(*this, cfg);
}

// ==========================================
// Quality Metrics (Delegated)
// ==========================================

real_t MeshBase::compute_quality(index_t cell, const std::string& metric) const {
    return MeshQuality::compute(*this, cell, metric);
}

std::pair<real_t,real_t> MeshBase::global_quality_range(const std::string& metric) const {
    return MeshQuality::global_range(*this, MeshQuality::metric_from_name(metric));
}

// ==========================================
// Adjacency Queries
// ==========================================

std::vector<index_t> MeshBase::cell_neighbors(index_t c) const {
    // Build cell2cell if not cached
    if (cell2cell_offsets_.empty()) {
        const_cast<MeshBase*>(this)->build_cell2cell();
    }

    std::vector<index_t> neighbors;
    if (c >= 0 && c < static_cast<index_t>(cell2cell_offsets_.size() - 1)) {
        offset_t start = cell2cell_offsets_[c];
        offset_t end = cell2cell_offsets_[c + 1];
        neighbors.reserve(end - start);
        for (offset_t i = start; i < end; ++i) {
            neighbors.push_back(cell2cell_[i]);
        }
    }

    return neighbors;
}

std::vector<index_t> MeshBase::vertex_cells(index_t n) const {
    // Build vertex2cell if not cached
    if (vertex2cell_offsets_.empty()) {
        const_cast<MeshBase*>(this)->build_vertex2cell();
    }

    std::vector<index_t> cells;
    if (n >= 0 && n < static_cast<index_t>(vertex2cell_offsets_.size() - 1)) {
        offset_t start = vertex2cell_offsets_[n];
        offset_t end = vertex2cell_offsets_[n + 1];
        cells.reserve(end - start);
        for (offset_t i = start; i < end; ++i) {
            cells.push_back(vertex2cell_[i]);
    }
    }

    return cells;
}

std::vector<index_t> MeshBase::face_neighbors(index_t f) const {
    std::vector<index_t> neighbors;
    if (f >= 0 && f < static_cast<index_t>(face2cell_.size())) {
        const auto& fc = face2cell_[f];
        if (fc[0] != INVALID_INDEX) neighbors.push_back(fc[0]);
        if (fc[1] != INVALID_INDEX) neighbors.push_back(fc[1]);
    }
    return neighbors;
}

std::vector<index_t> MeshBase::boundary_faces() const {
    std::vector<index_t> bfaces;
    for (index_t f = 0; f < static_cast<index_t>(face2cell_.size()); ++f) {
        if (face2cell_[f][1] == INVALID_INDEX) {
            bfaces.push_back(f);
        }
    }
    return bfaces;
}

std::vector<index_t> MeshBase::boundary_cells() const {
    std::unordered_set<index_t> bcells_set;
    for (index_t f = 0; f < static_cast<index_t>(face2cell_.size()); ++f) {
        if (face2cell_[f][1] == INVALID_INDEX && face2cell_[f][0] != INVALID_INDEX) {
            bcells_set.insert(face2cell_[f][0]);
        }
    }
    return std::vector<index_t>(bcells_set.begin(), bcells_set.end());
}

void MeshBase::build_vertex2cell() {
    MeshTopology::build_vertex2volume(*this, vertex2cell_offsets_, vertex2cell_);
}

void MeshBase::build_vertex2face() {
    MeshTopology::build_vertex2codim1(*this, vertex2face_offsets_, vertex2face_);
}

void MeshBase::build_cell2cell() {
    MeshTopology::build_cell2cell(*this, cell2cell_offsets_, cell2cell_);
}

// ==========================================
// Submesh Extraction
// ==========================================

MeshBase MeshBase::extract_submesh_by_region(label_t region_label) const {
    return extract_submesh_by_regions({region_label});
}

MeshBase MeshBase::extract_submesh_by_regions(const std::vector<label_t>& region_labels) const {
    // Create set of target labels for fast lookup
    std::unordered_set<label_t> target_labels(region_labels.begin(), region_labels.end());

    // Find cells with target labels
    std::vector<index_t> selected_cells;
    for (index_t c = 0; c < static_cast<index_t>(cell_region_id_.size()); ++c) {
        if (target_labels.count(cell_region_id_[c]) > 0) {
            selected_cells.push_back(c);
        }
    }

    // Extract unique vertices
    std::unordered_set<index_t> selected_vertices_set;
    for (index_t c : selected_cells) {
        auto [vertices_ptr, n_vertices] = cell_vertices_span(c);
        for (size_t i = 0; i < n_vertices; ++i) {
            selected_vertices_set.insert(vertices_ptr[i]);
        }
    }

    std::vector<index_t> selected_vertices(selected_vertices_set.begin(), selected_vertices_set.end());
    std::sort(selected_vertices.begin(), selected_vertices.end());

    // Build vertex renumbering map
    std::unordered_map<index_t, index_t> old2new_vertex;
    for (size_t i = 0; i < selected_vertices.size(); ++i) {
        old2new_vertex[selected_vertices[i]] = static_cast<index_t>(i);
    }

    // Build submesh coordinates
    std::vector<real_t> sub_X_ref;
    sub_X_ref.reserve(selected_vertices.size() * spatial_dim_);
    for (index_t n : selected_vertices) {
        for (int d = 0; d < spatial_dim_; ++d) {
            sub_X_ref.push_back(X_ref_[n * spatial_dim_ + d]);
        }
    }

    // Build submesh cells
    std::vector<offset_t> sub_cell2vertex_offsets;
    std::vector<index_t> sub_cell2vertex;
    std::vector<CellShape> sub_cell_shape;

    sub_cell2vertex_offsets.push_back(0);
    for (index_t c : selected_cells) {
        auto [vertices_ptr, n_vertices] = cell_vertices_span(c);
        for (size_t i = 0; i < n_vertices; ++i) {
            sub_cell2vertex.push_back(old2new_vertex[vertices_ptr[i]]);
        }
        sub_cell2vertex_offsets.push_back(static_cast<offset_t>(sub_cell2vertex.size()));
        sub_cell_shape.push_back(cell_shape_[c]);
    }

    // Create submesh
    MeshBase submesh;
    submesh.build_from_arrays(spatial_dim_, sub_X_ref, sub_cell2vertex_offsets,
                             sub_cell2vertex, sub_cell_shape);

    // Copy region labels
    for (size_t i = 0; i < selected_cells.size(); ++i) {
        submesh.set_region_label(static_cast<index_t>(i), cell_region_id_[selected_cells[i]]);
    }

    return submesh;
}

MeshBase MeshBase::extract_submesh_by_boundary(label_t boundary_label) const {
    // Find boundary faces with target label
    std::vector<index_t> boundary_faces_list = faces_with_label(boundary_label);

    // Collect unique cells adjacent to these faces
    std::unordered_set<index_t> selected_cells_set;
    for (index_t f : boundary_faces_list) {
        if (face2cell_[f][0] != INVALID_INDEX) {
            selected_cells_set.insert(face2cell_[f][0]);
        }
    }

    std::vector<index_t> selected_cells(selected_cells_set.begin(), selected_cells_set.end());

    // Extract as with regions (reuse logic)
    if (selected_cells.empty()) {
        return MeshBase();  // Return empty mesh
    }

    // Similar extraction logic as extract_submesh_by_regions
    // (code reuse - could factor out common extraction logic)
    std::unordered_set<index_t> selected_vertices_set;
    for (index_t c : selected_cells) {
        auto [vertices_ptr, n_vertices] = cell_vertices_span(c);
        for (size_t i = 0; i < n_vertices; ++i) {
            selected_vertices_set.insert(vertices_ptr[i]);
        }
    }

    std::vector<index_t> selected_vertices(selected_vertices_set.begin(), selected_vertices_set.end());
    std::sort(selected_vertices.begin(), selected_vertices.end());

    std::unordered_map<index_t, index_t> old2new_vertex;
    for (size_t i = 0; i < selected_vertices.size(); ++i) {
        old2new_vertex[selected_vertices[i]] = static_cast<index_t>(i);
    }

    std::vector<real_t> sub_X_ref;
    sub_X_ref.reserve(selected_vertices.size() * spatial_dim_);
    for (index_t n : selected_vertices) {
        for (int d = 0; d < spatial_dim_; ++d) {
            sub_X_ref.push_back(X_ref_[n * spatial_dim_ + d]);
        }
    }

    std::vector<offset_t> sub_cell2vertex_offsets;
    std::vector<index_t> sub_cell2vertex;
    std::vector<CellShape> sub_cell_shape;

    sub_cell2vertex_offsets.push_back(0);
    for (index_t c : selected_cells) {
        auto [vertices_ptr, n_vertices] = cell_vertices_span(c);
        for (size_t i = 0; i < n_vertices; ++i) {
            sub_cell2vertex.push_back(old2new_vertex[vertices_ptr[i]]);
        }
        sub_cell2vertex_offsets.push_back(static_cast<offset_t>(sub_cell2vertex.size()));
        sub_cell_shape.push_back(cell_shape_[c]);
    }

    MeshBase submesh;
    submesh.build_from_arrays(spatial_dim_, sub_X_ref, sub_cell2vertex_offsets,
                             sub_cell2vertex, sub_cell_shape);

    return submesh;
}

// ==========================================
// Search & Point Location
// ==========================================

PointLocateResult MeshBase::locate_point(const std::array<real_t,3>& x, Configuration cfg) const {
    return MeshSearch::locate_point(*this, x, cfg);
}

std::vector<PointLocateResult> MeshBase::locate_points(const std::vector<std::array<real_t,3>>& X,
                                                       Configuration cfg) const {
    return MeshSearch::locate_points(*this, X, cfg);
}

RayIntersectResult MeshBase::intersect_ray(const std::array<real_t,3>& origin,
                                          const std::array<real_t,3>& direction,
                                          Configuration cfg) const {
    return MeshSearch::intersect_ray(*this, origin, direction, cfg);
}

void MeshBase::build_search_structure(Configuration cfg) const {
    // In a full implementation, this would build acceleration structures
    // For now, just create the placeholder
    if (!search_accel_) {
        search_accel_ = std::make_unique<SearchAccel>();
    }
}

void MeshBase::clear_search_structure() const {
    search_accel_.reset();
}

// ==========================================
// Validation & Diagnostics
// ==========================================

void MeshBase::validate_basic() const {
    MeshValidation::validate_basic(*this);
}

void MeshBase::validate_topology() const {
    MeshValidation::validate_topology(*this);
}

void MeshBase::validate_geometry() const {
    MeshValidation::validate_geometry(*this);
}

void MeshBase::report_statistics() const {
    std::cout << "Mesh Statistics:\n";
    std::cout << "  Dimension: " << spatial_dim_ << "\n";
    std::cout << "  Vertices: " << n_vertices() << "\n";
    std::cout << "  Cells: " << n_cells() << "\n";
    std::cout << "  Faces: " << n_faces() << "\n";
    std::cout << "  Edges: " << n_edges() << "\n";

    if (!cell_shape_.empty()) {
        std::unordered_map<CellFamily, size_t> shape_counts;
        for (const auto& shape : cell_shape_) {
            shape_counts[shape.family]++;
        }
        std::cout << "  Cell types:\n";
        for (const auto& [family, count] : shape_counts) {
            std::cout << "    " << static_cast<int>(family) << ": " << count << "\n";
        }
    }

    auto bbox = bounding_box();
    std::cout << "  Bounding box:\n";
    std::cout << "    Min: [" << bbox.min[0] << ", " << bbox.min[1] << ", " << bbox.min[2] << "]\n";
    std::cout << "    Max: [" << bbox.max[0] << ", " << bbox.max[1] << ", " << bbox.max[2] << "]\n";
}

void MeshBase::write_debug(const std::string& prefix, const std::string& format) const {
    // Delegate to MeshIO when available
    // For now, just print statistics
    std::cout << "Debug output for " << prefix << " (format: " << format << ")\n";
    report_statistics();
}

// ==========================================
// Memory Management
// ==========================================

void MeshBase::shrink_to_fit() {
    X_ref_.shrink_to_fit();
    X_cur_.shrink_to_fit();

    cell_shape_.shrink_to_fit();
    cell2vertex_offsets_.shrink_to_fit();
    cell2vertex_.shrink_to_fit();

    face_shape_.shrink_to_fit();
    face2vertex_offsets_.shrink_to_fit();
    face2vertex_.shrink_to_fit();
    face2cell_.shrink_to_fit();

    edge2vertex_.shrink_to_fit();

    vertex_gid_.shrink_to_fit();
    cell_gid_.shrink_to_fit();
    face_gid_.shrink_to_fit();
    edge_gid_.shrink_to_fit();

    vertex_owner_.shrink_to_fit();
    cell_owner_.shrink_to_fit();
    face_owner_.shrink_to_fit();
    edge_owner_.shrink_to_fit();

    cell_region_id_.shrink_to_fit();
    face_boundary_id_.shrink_to_fit();

    vertex2cell_offsets_.shrink_to_fit();
    vertex2cell_.shrink_to_fit();
    vertex2face_offsets_.shrink_to_fit();
    vertex2face_.shrink_to_fit();
    cell2cell_offsets_.shrink_to_fit();
    cell2cell_.shrink_to_fit();
}

size_t MeshBase::memory_usage_bytes() const {
    size_t total = 0;

    // Coordinates
    total += X_ref_.capacity() * sizeof(real_t);
    total += X_cur_.capacity() * sizeof(real_t);

    // Cell topology
    total += cell_shape_.capacity() * sizeof(CellShape);
    total += cell2vertex_offsets_.capacity() * sizeof(offset_t);
    total += cell2vertex_.capacity() * sizeof(index_t);

    // Face topology
    total += face_shape_.capacity() * sizeof(CellShape);
    total += face2vertex_offsets_.capacity() * sizeof(offset_t);
    total += face2vertex_.capacity() * sizeof(index_t);
    total += face2cell_.capacity() * sizeof(std::array<index_t,2>);

    // Edge topology
    total += edge2vertex_.capacity() * sizeof(std::array<index_t,2>);

    // Global IDs
    total += vertex_gid_.capacity() * sizeof(gid_t);
    total += cell_gid_.capacity() * sizeof(gid_t);
    total += face_gid_.capacity() * sizeof(gid_t);
    total += edge_gid_.capacity() * sizeof(gid_t);

    // Ownership
    total += vertex_owner_.capacity() * sizeof(Ownership);
    total += cell_owner_.capacity() * sizeof(Ownership);
    total += face_owner_.capacity() * sizeof(Ownership);
    total += edge_owner_.capacity() * sizeof(Ownership);

    // Labels
    total += cell_region_id_.capacity() * sizeof(label_t);
    total += face_boundary_id_.capacity() * sizeof(label_t);

    // Adjacency caches
    total += vertex2cell_offsets_.capacity() * sizeof(offset_t);
    total += vertex2cell_.capacity() * sizeof(index_t);
    total += vertex2face_offsets_.capacity() * sizeof(offset_t);
    total += vertex2face_.capacity() * sizeof(index_t);
    total += cell2cell_offsets_.capacity() * sizeof(offset_t);
    total += cell2cell_.capacity() * sizeof(index_t);

    // Field attachments
    for (int i = 0; i < 4; ++i) {
        for (const auto& [name, info] : attachments_[i].by_name) {
            total += info.data.capacity();
        }
    }

    return total;
}

// ==========================================
// IO Registry
// ==========================================

std::unordered_map<std::string, MeshBase::LoadFn>& MeshBase::readers_() {
    static std::unordered_map<std::string, LoadFn> registry;
    return registry;
}

std::unordered_map<std::string, MeshBase::SaveFn>& MeshBase::writers_() {
    static std::unordered_map<std::string, SaveFn> registry;
    return registry;
}

void MeshBase::register_reader(const std::string& format, LoadFn fn) {
    readers_()[format] = fn;
}

void MeshBase::register_writer(const std::string& format, SaveFn fn) {
    writers_()[format] = fn;
}

MeshBase MeshBase::load(const MeshIOOptions& opts) {
    auto it = readers_().find(opts.format);
    if (it == readers_().end()) {
        throw std::runtime_error("MeshBase::load: no reader for format '" + opts.format + "'");
    }
    return it->second(opts);
}

void MeshBase::save(const MeshIOOptions& opts) const {
    auto it = writers_().find(opts.format);
    if (it == writers_().end()) {
        throw std::runtime_error("MeshBase::save: no writer for format '" + opts.format + "'");
    }
    it->second(*this, opts);
}

std::vector<std::string> MeshBase::registered_readers() {
    std::vector<std::string> formats;
    for (const auto& [format, fn] : readers_()) {
        formats.push_back(format);
    }
    std::sort(formats.begin(), formats.end());
    return formats;
}

std::vector<std::string> MeshBase::registered_writers() {
    std::vector<std::string> formats;
    for (const auto& [format, fn] : writers_()) {
        formats.push_back(format);
    }
    std::sort(formats.begin(), formats.end());
    return formats;
}

// ==========================================
// Helper Methods
// ==========================================

size_t MeshBase::entity_count(EntityKind k) const noexcept {
    switch (k) {
        case EntityKind::Vertex: return n_vertices();
        case EntityKind::Edge:   return n_edges();
        case EntityKind::Face:   return n_faces();
        case EntityKind::Volume: return n_cells();
    }
    return 0;
}

void MeshBase::invalidate_caches() {
    vertex2cell_offsets_.clear();
    vertex2cell_.clear();
    vertex2face_offsets_.clear();
    vertex2face_.clear();
    cell2cell_offsets_.clear();
    cell2cell_.clear();
}

// ---- GID setters and rebuilders ----

void MeshBase::set_vertex_gids(std::vector<gid_t> gids) {
    vertex_gid_ = std::move(gids);
    // Rebuild vertex map only
    global2local_vertex_.clear();
    for (index_t i = 0; i < static_cast<index_t>(vertex_gid_.size()); ++i) {
        global2local_vertex_[vertex_gid_[i]] = i;
    }
}

void MeshBase::set_cell_gids(std::vector<gid_t> gids) {
    cell_gid_ = std::move(gids);
    // Rebuild cell map only
    global2local_cell_.clear();
    for (index_t i = 0; i < static_cast<index_t>(cell_gid_.size()); ++i) {
        global2local_cell_[cell_gid_[i]] = i;
    }
}

void MeshBase::set_face_gids(std::vector<gid_t> gids) {
    face_gid_ = std::move(gids);
    // Rebuild face map only
    global2local_face_.clear();
    for (index_t i = 0; i < static_cast<index_t>(face_gid_.size()); ++i) {
        global2local_face_[face_gid_[i]] = i;
    }
}

void MeshBase::set_edge_gids(std::vector<gid_t> gids) {
    edge_gid_ = std::move(gids);
    // Rebuild edge map only
    global2local_edge_.clear();
    for (index_t i = 0; i < static_cast<index_t>(edge_gid_.size()); ++i) {
        global2local_edge_[edge_gid_[i]] = i;
    }
}

void MeshBase::rebuild_gid_maps() {
    global2local_vertex_.clear();
    for (index_t i = 0; i < static_cast<index_t>(vertex_gid_.size()); ++i) {
        global2local_vertex_[vertex_gid_[i]] = i;
    }
    global2local_cell_.clear();
    for (index_t i = 0; i < static_cast<index_t>(cell_gid_.size()); ++i) {
        global2local_cell_[cell_gid_[i]] = i;
    }
    global2local_face_.clear();
    for (index_t i = 0; i < static_cast<index_t>(face_gid_.size()); ++i) {
        global2local_face_[face_gid_[i]] = i;
    }
    global2local_edge_.clear();
    for (index_t i = 0; i < static_cast<index_t>(edge_gid_.size()); ++i) {
        global2local_edge_[edge_gid_[i]] = i;
    }
}

} // namespace svmp
