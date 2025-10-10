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

#include "Mesh.h"
#include "SpatialHashing.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

namespace svmp {

// ========================
// CellShapeRegistry implementation
// ========================

std::unordered_map<std::string, std::unordered_map<int, CellShape>>& CellShapeRegistry::map_() {
  static std::unordered_map<std::string, std::unordered_map<int, CellShape>> registry;
  return registry;
}

void CellShapeRegistry::register_shape(const std::string& format, int type_id, const CellShape& shape) {
  map_()[format][type_id] = shape;
}

bool CellShapeRegistry::has(const std::string& format, int type_id) {
  auto it_format = map_().find(format);
  if (it_format == map_().end()) return false;
  return it_format->second.find(type_id) != it_format->second.end();
}

CellShape CellShapeRegistry::get(const std::string& format, int type_id) {
  auto it_format = map_().find(format);
  if (it_format == map_().end()) {
    throw std::runtime_error("No cell shapes registered for format: " + format);
  }
  auto it_shape = it_format->second.find(type_id);
  if (it_shape == it_format->second.end()) {
    throw std::runtime_error("CellShape not registered for format '" + format +
                            "' type ID " + std::to_string(type_id));
  }
  return it_shape->second;
}

void CellShapeRegistry::clear_format(const std::string& format) {
  map_().erase(format);
}

// ========================
// Search acceleration structure
// ========================
struct MeshBase::SearchAccel {
  // Simple AABB tree placeholder - in production would use a proper BVH library
  struct AABB {
    std::array<real_t,3> min = {{+1e300, +1e300, +1e300}};
    std::array<real_t,3> max = {{-1e300, -1e300, -1e300}};

    void expand(const std::array<real_t,3>& point) {
      for (int i = 0; i < 3; ++i) {
        min[i] = std::min(min[i], point[i]);
        max[i] = std::max(max[i], point[i]);
      }
    }

    bool contains(const std::array<real_t,3>& point) const {
      for (int i = 0; i < 3; ++i) {
        if (point[i] < min[i] || point[i] > max[i]) return false;
      }
      return true;
    }
  };

  std::vector<AABB> cell_boxes;
  AABB global_box;
  Configuration built_for_config = Configuration::Reference;

  void build(const MeshBase& mesh, Configuration cfg) {
    cell_boxes.resize(mesh.n_cells());
    global_box = AABB{};
    built_for_config = cfg;

    const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                        ? mesh.X_cur() : mesh.X_ref();

    for (size_t c = 0; c < mesh.n_cells(); ++c) {
      auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(static_cast<index_t>(c));
      AABB& box = cell_boxes[c];

      for (size_t i = 0; i < n_nodes; ++i) {
        index_t node_id = nodes_ptr[i];
        std::array<real_t,3> pt = {{0,0,0}};
        for (int d = 0; d < mesh.dim(); ++d) {
          pt[d] = coords[node_id * mesh.dim() + d];
        }
        box.expand(pt);
        global_box.expand(pt);
      }
    }
  }
};

// ========================
// MeshBase implementation
// ========================

// ---- Constructors & lifecycle
MeshBase::MeshBase() : spatial_dim_(0) {
}

MeshBase::MeshBase(int spatial_dim) : spatial_dim_(spatial_dim) {
  if (spatial_dim < 1 || spatial_dim > 3) {
    throw std::invalid_argument("MeshBase: spatial_dim must be 1, 2, or 3");
  }
}

void MeshBase::clear() {
  spatial_dim_ = 0;
  X_ref_.clear();
  X_cur_.clear();

  cell_shape_.clear();
  cell2node_offsets_.clear();
  cell2node_.clear();

  face_shape_.clear();
  face2node_offsets_.clear();
  face2node_.clear();
  face2cell_.clear();

  edge2node_.clear();
  edge_gid_.clear();
  edge_owner_.clear();

  node_gid_.clear();
  face_gid_.clear();
  cell_gid_.clear();
  node_owner_.clear();
  face_owner_.clear();
  cell_owner_.clear();

  cell_region_id_.clear();
  face_boundary_id_.clear();
  for (int i = 0; i < 4; ++i) {
    entity_sets_[i].clear();
    attachments_[i].by_name.clear();
  }

  cell_parent_.clear();
  cell_children_offsets_.clear();
  cell_children_.clear();
  refine_flag_.clear();
  coarsen_flag_.clear();

  global2local_cell_.clear();
  global2local_node_.clear();
  global2local_face_.clear();

  node2cell_offsets_.clear();
  node2cell_.clear();
  node2face_offsets_.clear();
  node2face_.clear();
  cell2cell_offsets_.clear();
  cell2cell_.clear();

  cell_geom_order_.clear();
  cell2geomnode_offsets_.clear();
  cell2geomnode_.clear();

  label_from_name_.clear();
  name_from_label_.clear();

  cell_measures_.clear();
  face_areas_.clear();
  face_normals_.clear();
  geom_cache_valid_ = false;

  periodic_pairs_.clear();
  mortar_interfaces_.clear();
  contact_candidates_.clear();

  search_accel_.reset();

  next_field_id_ = 1;
  field_index_.clear();
}

void MeshBase::reserve(index_t n_nodes, index_t n_cells, index_t n_faces) {
  X_ref_.reserve(n_nodes * spatial_dim_);
  cell_shape_.reserve(n_cells);
  cell2node_offsets_.reserve(n_cells + 1);
  if (n_faces > 0) {
    face_shape_.reserve(n_faces);
    face2node_offsets_.reserve(n_faces + 1);
    face2cell_.reserve(n_faces);
  }
}

void MeshBase::build_from_arrays(
    int spatial_dim,
    const std::vector<real_t>& X_ref,
    const std::vector<offset_t>& cell2node_offsets,
    const std::vector<index_t>& cell2node,
    const std::vector<CellShape>& cell_shape) {

  clear();
  spatial_dim_ = spatial_dim;

  if (X_ref.size() % spatial_dim != 0) {
    throw std::invalid_argument("X_ref size must be divisible by spatial_dim");
  }

  size_t n_cells = cell_shape.size();
  if (cell2node_offsets.size() != n_cells + 1) {
    throw std::invalid_argument("cell2node_offsets size must be n_cells + 1");
  }

  X_ref_ = X_ref;
  cell_shape_ = cell_shape;
  cell2node_offsets_ = cell2node_offsets;
  cell2node_ = cell2node;

  // Initialize region labels to 0
  cell_region_id_.resize(n_cells, 0);

  // Initialize parent to -1 (no parent)
  cell_parent_.resize(n_cells, -1);

  // Initialize ownership to Owned
  size_t n_nodes = X_ref.size() / spatial_dim;
  node_owner_.resize(n_nodes, Ownership::Owned);
  cell_owner_.resize(n_cells, Ownership::Owned);
}

void MeshBase::set_faces_from_arrays(
    const std::vector<CellShape>& face_shape,
    const std::vector<offset_t>& face2node_offsets,
    const std::vector<index_t>& face2node,
    const std::vector<std::array<index_t,2>>& face2cell) {

  size_t n_faces = face_shape.size();
  if (face2node_offsets.size() != n_faces + 1) {
    throw std::invalid_argument("face2node_offsets size must be n_faces + 1");
  }
  if (face2cell.size() != n_faces) {
    throw std::invalid_argument("face2cell size must match n_faces");
  }

  face_shape_ = face_shape;
  face2node_offsets_ = face2node_offsets;
  face2node_ = face2node;
  face2cell_ = face2cell;

  // Initialize boundary labels to 0
  face_boundary_id_.resize(n_faces, 0);
  face_owner_.resize(n_faces, Ownership::Owned);
}

void MeshBase::set_edges_from_arrays(const std::vector<std::array<index_t,2>>& edge2node) {
  edge2node_ = edge2node;
  edge_owner_.resize(edge2node.size(), Ownership::Owned);
}

void MeshBase::finalize() {
  // Sanity check GID arrays match entity counts
  if (!node_gid_.empty() && node_gid_.size() != n_nodes()) {
    throw std::runtime_error("finalize(): node_gid size (" + std::to_string(node_gid_.size()) +
                           ") doesn't match n_nodes (" + std::to_string(n_nodes()) + ")");
  }
  if (!cell_gid_.empty() && cell_gid_.size() != n_cells()) {
    throw std::runtime_error("finalize(): cell_gid size (" + std::to_string(cell_gid_.size()) +
                           ") doesn't match n_cells (" + std::to_string(n_cells()) + ")");
  }
  if (!face_gid_.empty() && face_gid_.size() != n_faces()) {
    throw std::runtime_error("finalize(): face_gid size (" + std::to_string(face_gid_.size()) +
                           ") doesn't match n_faces (" + std::to_string(n_faces()) + ")");
  }

  // Build global ID maps
  for (size_t i = 0; i < node_gid_.size(); ++i) {
    global2local_node_[node_gid_[i]] = static_cast<index_t>(i);
  }
  for (size_t i = 0; i < cell_gid_.size(); ++i) {
    global2local_cell_[cell_gid_[i]] = static_cast<index_t>(i);
  }
  for (size_t i = 0; i < face_gid_.size(); ++i) {
    global2local_face_[face_gid_[i]] = static_cast<index_t>(i);
  }

  // Basic validation
  validate_basic();

  // Clear geometry caches to be rebuilt on demand
  invalidate_caches();

  // Notify observers that topology has been established
  event_bus_.notify(MeshEvent::TopologyChanged);
}

// ---- Topology access
std::pair<const index_t*, size_t> MeshBase::cell_nodes_span(index_t c) const {
  if (c < 0 || static_cast<size_t>(c) >= cell_shape_.size()) {
    throw std::out_of_range("Cell index out of bounds");
  }

  offset_t start = cell2node_offsets_[c];
  offset_t end = cell2node_offsets_[c + 1];
  size_t count = static_cast<size_t>(end - start);

  return {&cell2node_[start], count};
}

std::pair<const index_t*, size_t> MeshBase::face_nodes_span(index_t f) const {
  if (f < 0 || static_cast<size_t>(f) >= face_shape_.size()) {
    throw std::out_of_range("Face index out of bounds");
  }

  offset_t start = face2node_offsets_[f];
  offset_t end = face2node_offsets_[f + 1];
  size_t count = static_cast<size_t>(end - start);

  return {&face2node_[start], count};
}

// ---- Coordinates
void MeshBase::set_current_coords(const std::vector<real_t>& Xcur) {
  if (Xcur.size() != X_ref_.size()) {
    throw std::invalid_argument("Current coords size must match reference coords");
  }
  X_cur_ = Xcur;
  invalidate_caches();
  event_bus_.notify(MeshEvent::GeometryChanged);
}

void MeshBase::clear_current_coords() {
  X_cur_.clear();
  invalidate_caches();
  event_bus_.notify(MeshEvent::GeometryChanged);
}

// ---- Labels & sets
void MeshBase::set_region_label(index_t cell, label_t label) {
  if (cell < 0 || static_cast<size_t>(cell) >= cell_shape_.size()) {
    throw std::out_of_range("Cell index out of bounds");
  }
  if (cell_region_id_.size() != cell_shape_.size()) {
    cell_region_id_.resize(cell_shape_.size(), 0);
  }
  cell_region_id_[cell] = label;
  event_bus_.notify(MeshEvent::LabelsChanged);
}

label_t MeshBase::region_label(index_t cell) const {
  if (cell < 0 || static_cast<size_t>(cell) >= cell_shape_.size()) {
    throw std::out_of_range("Cell index out of bounds");
  }
  return cell_region_id_[cell];
}

std::vector<index_t> MeshBase::cells_with_label(label_t label) const {
  std::vector<index_t> result;
  for (size_t i = 0; i < cell_region_id_.size(); ++i) {
    if (cell_region_id_[i] == label) {
      result.push_back(static_cast<index_t>(i));
    }
  }
  return result;
}

void MeshBase::set_boundary_label(index_t face, label_t label) {
  if (face < 0 || static_cast<size_t>(face) >= face_shape_.size()) {
    throw std::out_of_range("Face index out of bounds");
  }
  if (face_boundary_id_.size() != face_shape_.size()) {
    face_boundary_id_.resize(face_shape_.size(), 0);
  }
  face_boundary_id_[face] = label;
  event_bus_.notify(MeshEvent::LabelsChanged);
}

label_t MeshBase::boundary_label(index_t face) const {
  if (face < 0 || static_cast<size_t>(face) >= face_shape_.size()) {
    throw std::out_of_range("Face index out of bounds");
  }
  return face_boundary_id_[face];
}

std::vector<index_t> MeshBase::faces_with_label(label_t label) const {
  std::vector<index_t> result;
  for (size_t i = 0; i < face_boundary_id_.size(); ++i) {
    if (face_boundary_id_[i] == label) {
      result.push_back(static_cast<index_t>(i));
    }
  }
  return result;
}

void MeshBase::add_to_set(EntityKind kind, const std::string& name, index_t id) {
  entity_sets_[static_cast<int>(kind)][name].push_back(id);
}

const std::vector<index_t>& MeshBase::get_set(EntityKind kind, const std::string& name) const {
  static const std::vector<index_t> empty;
  const auto& sets = entity_sets_[static_cast<int>(kind)];
  auto it = sets.find(name);
  return (it != sets.end()) ? it->second : empty;
}

bool MeshBase::has_set(EntityKind kind, const std::string& name) const {
  const auto& sets = entity_sets_[static_cast<int>(kind)];
  return sets.find(name) != sets.end();
}

// ---- Label <-> name registry
void MeshBase::register_label(const std::string& name, label_t label) {
  label_from_name_[name] = label;
  if (label >= 0) {
    if (static_cast<size_t>(label) >= name_from_label_.size()) {
      name_from_label_.resize(label + 1);
    }
    name_from_label_[label] = name;
  }
}

std::string MeshBase::label_name(label_t label) const {
  if (label >= 0 && static_cast<size_t>(label) < name_from_label_.size()) {
    return name_from_label_[label];
  }
  return "";
}

label_t MeshBase::label_from_name(const std::string& name) const {
  auto it = label_from_name_.find(name);
  return (it != label_from_name_.end()) ? it->second : -1;
}

// ---- Attachment system
MeshBase::FieldHandle MeshBase::attach_field(EntityKind kind, const std::string& name,
                                             FieldScalarType type, size_t components,
                                             size_t custom_bytes_per_component) {
  auto& attach_table = attachments_[static_cast<int>(kind)];

  // Check if field already exists
  if (attach_table.by_name.find(name) != attach_table.by_name.end()) {
    throw std::runtime_error("Field '" + name + "' already exists");
  }

  size_t n_entities = entity_count(kind);
  size_t bytes_per_comp = (type == FieldScalarType::Custom) ? custom_bytes_per_component : bytes_per(type);

  if (type == FieldScalarType::Custom && custom_bytes_per_component == 0) {
    throw std::invalid_argument("Custom field type requires non-zero bytes_per_component");
  }

  size_t bytes = bytes_per_comp * components * n_entities;

  FieldInfo info;
  info.type = type;
  info.components = components;
  info.bytes_per_component = bytes_per_comp;
  info.data.resize(bytes, 0);

  attach_table.by_name[name] = std::move(info);

  FieldHandle handle;
  handle.id = next_field_id_++;
  handle.kind = kind;
  handle.name = name;

  field_index_[handle.id] = {kind, name};

  event_bus_.notify(MeshEvent::FieldsChanged);
  return handle;
}

MeshBase::FieldHandle MeshBase::attach_field_with_descriptor(EntityKind kind, const std::string& name,
                                                            FieldScalarType type,
                                                            const FieldDescriptor& descriptor) {
  // First attach the field using the regular method
  auto handle = attach_field(kind, name, type, descriptor.components);

  // Store the descriptor
  field_descriptors_[handle.id] = descriptor;

  return handle;
}

bool MeshBase::has_field(EntityKind kind, const std::string& name) const {
  const auto& attach_table = attachments_[static_cast<int>(kind)];
  return attach_table.by_name.find(name) != attach_table.by_name.end();
}

void MeshBase::remove_field(const FieldHandle& h) {
  auto it = field_index_.find(h.id);
  if (it == field_index_.end()) return;

  auto [kind, name] = it->second;
  auto& attach_table = attachments_[static_cast<int>(kind)];
  attach_table.by_name.erase(name);
  field_index_.erase(it);
}

void* MeshBase::field_data(const FieldHandle& h) {
  auto it = field_index_.find(h.id);
  if (it == field_index_.end()) return nullptr;

  auto [kind, name] = it->second;
  auto& attach_table = attachments_[static_cast<int>(kind)];
  auto field_it = attach_table.by_name.find(name);
  if (field_it == attach_table.by_name.end()) return nullptr;

  return field_it->second.data.data();
}

const void* MeshBase::field_data(const FieldHandle& h) const {
  auto it = field_index_.find(h.id);
  if (it == field_index_.end()) return nullptr;

  auto [kind, name] = it->second;
  const auto& attach_table = attachments_[static_cast<int>(kind)];
  auto field_it = attach_table.by_name.find(name);
  if (field_it == attach_table.by_name.end()) return nullptr;

  return field_it->second.data.data();
}

size_t MeshBase::field_components(const FieldHandle& h) const {
  auto it = field_index_.find(h.id);
  if (it == field_index_.end()) return 0;

  auto [kind, name] = it->second;
  const auto& attach_table = attachments_[static_cast<int>(kind)];
  auto field_it = attach_table.by_name.find(name);
  if (field_it == attach_table.by_name.end()) return 0;

  return field_it->second.components;
}

FieldScalarType MeshBase::field_type(const FieldHandle& h) const {
  auto it = field_index_.find(h.id);
  if (it == field_index_.end()) return FieldScalarType::Custom;

  auto [kind, name] = it->second;
  const auto& attach_table = attachments_[static_cast<int>(kind)];
  auto field_it = attach_table.by_name.find(name);
  if (field_it == attach_table.by_name.end()) return FieldScalarType::Custom;

  return field_it->second.type;
}

size_t MeshBase::field_entity_count(const FieldHandle& h) const {
  auto it = field_index_.find(h.id);
  if (it == field_index_.end()) return 0;

  auto [kind, name] = it->second;
  return entity_count(kind);
}

size_t MeshBase::field_bytes_per_entity(const FieldHandle& h) const {
  auto it = field_index_.find(h.id);
  if (it == field_index_.end()) return 0;

  auto [kind, name] = it->second;
  const auto& attach_table = attachments_[static_cast<int>(kind)];
  auto field_it = attach_table.by_name.find(name);
  if (field_it == attach_table.by_name.end()) return 0;

  return field_it->second.bytes_per_component * field_it->second.components;
}

// ---- Geometry helpers
std::array<real_t,3> MeshBase::cell_center(index_t c, Configuration cfg) const {
  auto [nodes_ptr, n_nodes] = cell_nodes_span(c);

  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = (cfg == Configuration::Current && !X_cur_.empty())
                                      ? X_cur_ : X_ref_;

  for (size_t i = 0; i < n_nodes; ++i) {
    index_t node_id = nodes_ptr[i];
    for (int d = 0; d < spatial_dim_; ++d) {
      center[d] += coords[node_id * spatial_dim_ + d];
    }
  }

  if (n_nodes > 0) {
    for (int d = 0; d < 3; ++d) {
      center[d] /= n_nodes;
    }
  }

  return center;
}

std::array<real_t,3> MeshBase::face_center(index_t f, Configuration cfg) const {
  auto [nodes_ptr, n_nodes] = face_nodes_span(f);

  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = (cfg == Configuration::Current && !X_cur_.empty())
                                      ? X_cur_ : X_ref_;

  for (size_t i = 0; i < n_nodes; ++i) {
    index_t node_id = nodes_ptr[i];
    for (int d = 0; d < spatial_dim_; ++d) {
      center[d] += coords[node_id * spatial_dim_ + d];
    }
  }

  if (n_nodes > 0) {
    for (int d = 0; d < 3; ++d) {
      center[d] /= n_nodes;
    }
  }

  return center;
}

MeshBase::BoundingBox MeshBase::bounding_box(Configuration cfg) const {
  BoundingBox box;

  const std::vector<real_t>& coords = (cfg == Configuration::Current && !X_cur_.empty())
                                      ? X_cur_ : X_ref_;

  size_t n_nodes = coords.size() / spatial_dim_;
  for (size_t i = 0; i < n_nodes; ++i) {
    for (int d = 0; d < spatial_dim_; ++d) {
      real_t val = coords[i * spatial_dim_ + d];
      box.min[d] = std::min(box.min[d], val);
      box.max[d] = std::max(box.max[d], val);
    }
  }

  return box;
}

// ---- Reordering
void MeshBase::reorder(ReorderAlgo algo) {
  // Placeholder for reordering algorithms
  // In production would implement RCM, Hilbert, Morton, etc.
  if (algo != ReorderAlgo::None) {
    // Future implementation
  }
}

// ---- Validation & diagnostics
void MeshBase::validate_basic() const {
  // Check cell connectivity
  if (!cell2node_offsets_.empty()) {
    if (cell2node_offsets_.size() != cell_shape_.size() + 1) {
      throw std::runtime_error("Cell offsets size mismatch");
    }
    if (cell2node_offsets_.back() != static_cast<index_t>(cell2node_.size())) {
      throw std::runtime_error("Cell connectivity size mismatch");
    }
  }

  // Check face connectivity
  if (!face2node_offsets_.empty()) {
    if (face2node_offsets_.size() != face_shape_.size() + 1) {
      throw std::runtime_error("Face offsets size mismatch");
    }
    if (face2node_offsets_.back() != static_cast<index_t>(face2node_.size())) {
      throw std::runtime_error("Face connectivity size mismatch");
    }
  }

  // Check node coordinates
  if (X_ref_.size() % spatial_dim_ != 0) {
    throw std::runtime_error("Coordinate array size not divisible by spatial dimension");
  }
}

void MeshBase::validate_topology() const {
  // Use robust spatial hashing for O(N) complexity validation
  const real_t tol = 1e-10;
  if (n_nodes() == 0) return;

  // Create mesh validator with spatial hashing
  validation::MeshValidator validator(const_cast<MeshBase&>(*this), tol);

  // Check for duplicate nodes
  auto duplicates = validator.find_duplicate_nodes();
  if (!duplicates.empty()) {
    for (const auto& [n1, n2] : duplicates) {
      std::cerr << "Warning: Duplicate nodes " << n1 << " and " << n2
                << " are within tolerance " << tol << std::endl;
    }
  }

  // Check for inverted elements
  auto inverted = validator.find_inverted_elements();
  for (index_t c : inverted) {
    std::cerr << "Warning: Cell " << c << " appears to be inverted (negative Jacobian)" << std::endl;
  }

  // Check for degenerate elements
  auto degenerate = validator.find_degenerate_elements(tol);
  for (index_t c : degenerate) {
    real_t measure = cell_measure(c);
    std::cerr << "Warning: Cell " << c << " appears to be degenerate (measure = " << measure << ")" << std::endl;
  }

  // Check for repeated nodes within cells
  for (size_t c = 0; c < n_cells(); ++c) {
    auto [nodes_ptr, n_nodes] = cell_nodes_span(static_cast<index_t>(c));

    std::unordered_set<index_t> unique_nodes;
    for (size_t i = 0; i < n_nodes; ++i) {
      if (!unique_nodes.insert(nodes_ptr[i]).second) {
        std::cerr << "Warning: Cell " << c << " has repeated node " << nodes_ptr[i] << std::endl;
      }
    }
  }
}

void MeshBase::validate_geometry() const {
  // Check face orientation consistency
  // Check outward normals
  // This would require shape-specific logic
}

void MeshBase::report_statistics() const {
  std::cout << "Mesh Statistics:" << std::endl;
  std::cout << "  Dimension: " << spatial_dim_ << std::endl;
  std::cout << "  Nodes: " << n_nodes() << std::endl;
  std::cout << "  Cells: " << n_cells() << std::endl;
  std::cout << "  Faces: " << n_faces() << std::endl;
  std::cout << "  Edges: " << n_edges() << std::endl;

  auto bbox = bounding_box();
  std::cout << "  Bounding Box: ["
            << bbox.min[0] << ", " << bbox.max[0] << "] x ["
            << bbox.min[1] << ", " << bbox.max[1] << "] x ["
            << bbox.min[2] << ", " << bbox.max[2] << "]" << std::endl;

  // Use comprehensive validation report with spatial hashing
  validation::MeshValidator validator(const_cast<MeshBase&>(*this), 1e-10);
  validator.validate_and_report();
}

void MeshBase::write_debug(const std::string& prefix, const std::string& format) const {
  // Format-agnostic debug output - delegates to registered IO writers or uses simple default

  // Helper to map CellFamily to VTK type IDs for VTK formats
  auto cell_family_to_vtk_type = [](CellFamily fam, int order) -> int {
    // Linear elements (order 1)
    if (order == 1) {
      switch (fam) {
        case CellFamily::Line:      return 3;   // VTK_LINE
        case CellFamily::Triangle:  return 5;   // VTK_TRIANGLE
        case CellFamily::Quad:      return 9;   // VTK_QUAD
        case CellFamily::Tetra:     return 10;  // VTK_TETRA
        case CellFamily::Hex:       return 12;  // VTK_HEXAHEDRON
        case CellFamily::Wedge:     return 13;  // VTK_WEDGE
        case CellFamily::Pyramid:   return 14;  // VTK_PYRAMID
        case CellFamily::Polygon:   return 7;   // VTK_POLYGON
        case CellFamily::Polyhedron: return 42; // VTK_POLYHEDRON
      }
    }
    // Quadratic elements (order 2)
    else if (order == 2) {
      switch (fam) {
        case CellFamily::Line:      return 21;  // VTK_QUADRATIC_EDGE
        case CellFamily::Triangle:  return 22;  // VTK_QUADRATIC_TRIANGLE
        case CellFamily::Quad:      return 23;  // VTK_QUADRATIC_QUAD
        case CellFamily::Tetra:     return 24;  // VTK_QUADRATIC_TETRA
        case CellFamily::Hex:       return 25;  // VTK_QUADRATIC_HEXAHEDRON
        case CellFamily::Wedge:     return 26;  // VTK_QUADRATIC_WEDGE
        case CellFamily::Pyramid:   return 27;  // VTK_QUADRATIC_PYRAMID
        default: break;
      }
    }
    return 7; // Default to polygon if unknown
  };

  if (format == "vtu" || format == "vtk") {
    // VTK/VTU ASCII writer with labels and ownership information
    std::string filename = prefix + "." + format;
    std::ofstream file(filename);

    file << "# vtk DataFile Version 3.0\n";
    file << "MeshBase debug output with labels and ownership\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // Write points
    size_t n_pts = n_nodes();
    file << "POINTS " << n_pts << " double\n";
    for (size_t i = 0; i < n_pts; ++i) {
      for (int d = 0; d < 3; ++d) {
        if (d < spatial_dim_) {
          file << X_ref_[i * spatial_dim_ + d] << " ";
        } else {
          file << "0.0 ";
        }
      }
      file << "\n";
    }

    // Write cells
    size_t total_conn_size = cell2node_.size() + n_cells();
    file << "CELLS " << n_cells() << " " << total_conn_size << "\n";

    for (size_t c = 0; c < n_cells(); ++c) {
      auto [nodes_ptr, n_nodes_cell] = cell_nodes_span(static_cast<index_t>(c));
      file << n_nodes_cell;
      for (size_t i = 0; i < n_nodes_cell; ++i) {
        file << " " << nodes_ptr[i];
      }
      file << "\n";
    }

    // Write cell types
    file << "CELL_TYPES " << n_cells() << "\n";
    for (size_t c = 0; c < n_cells(); ++c) {
      int vtk_type = cell_family_to_vtk_type(cell_shape_[c].family, cell_shape_[c].order);
      file << vtk_type << "\n";
    }

  // ===== Enhanced: Cell Data Section =====
  if (n_cells() > 0) {
    file << "\nCELL_DATA " << n_cells() << "\n";

    // Write region labels
    if (!cell_region_id_.empty()) {
      file << "SCALARS RegionLabel int 1\n";
      file << "LOOKUP_TABLE default\n";
      for (size_t c = 0; c < n_cells(); ++c) {
        file << (c < cell_region_id_.size() ? cell_region_id_[c] : 0) << "\n";
      }
    }

    // Write cell ownership (if available)
    if (!cell_owner_.empty()) {
      file << "SCALARS Ownership int 1\n";
      file << "LOOKUP_TABLE ownership_colors\n";
      for (size_t c = 0; c < n_cells(); ++c) {
        if (c < cell_owner_.size()) {
          // Map ownership to integer: Owned=0, Ghost=1, Shared=2
          file << static_cast<int>(cell_owner_[c]) << "\n";
        } else {
          file << "0\n";
        }
      }
    }

    // Write cell quality metrics
    file << "SCALARS CellQuality double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t c = 0; c < n_cells(); ++c) {
      real_t quality = compute_quality(static_cast<index_t>(c), "aspect_ratio");
      file << quality << "\n";
    }

    // Write cell measure (area/volume)
    file << "SCALARS CellMeasure double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t c = 0; c < n_cells(); ++c) {
      real_t measure = cell_measure(static_cast<index_t>(c));
      file << measure << "\n";
    }
  }

  // ===== Enhanced: Point Data Section =====
  if (n_pts > 0) {
    file << "\nPOINT_DATA " << n_pts << "\n";

    // Write node ownership (if available)
    if (!node_owner_.empty()) {
      file << "SCALARS NodeOwnership int 1\n";
      file << "LOOKUP_TABLE ownership_colors\n";
      for (size_t n = 0; n < n_pts; ++n) {
        if (n < node_owner_.size()) {
          // Map ownership to integer: Owned=0, Ghost=1, Shared=2
          file << static_cast<int>(node_owner_[n]) << "\n";
        } else {
          file << "0\n";
        }
      }
    }

    // Write node global IDs (if available)
    if (!node_gid_.empty()) {
      file << "SCALARS GlobalNodeID int 1\n";
      file << "LOOKUP_TABLE default\n";
      for (size_t n = 0; n < n_pts; ++n) {
        if (n < node_gid_.size()) {
          file << node_gid_[n] << "\n";
        } else {
          file << n << "\n";
        }
      }
    }
  }

    // ===== Enhanced: Lookup Tables for Ownership =====
    file << "\nLOOKUP_TABLE ownership_colors 3\n";
    file << "0.0 1.0 0.0 1.0\n";  // Owned: Green
    file << "1.0 0.0 0.0 1.0\n";  // Ghost: Red
    file << "0.0 0.0 1.0 1.0\n";  // Shared: Blue

    file.close();

    std::cout << "Debug mesh file written to: " << filename << std::endl;
    std::cout << "  Includes: region labels, ownership flags, quality metrics, measures" << std::endl;
  } else {
    // For other formats, use the registered writer if available
    MeshIOOptions opts;
    opts.format = format;
    opts.path = prefix + "." + format;
    opts.kv["debug"] = "true";

    try {
      save(opts);
      std::cout << "Debug mesh file written using '" << format << "' format: " << opts.path << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Warning: Unable to write debug file in format '" << format << "': " << e.what() << std::endl;
      std::cerr << "  Falling back to VTK format." << std::endl;
      // Fallback to VTK
      write_debug(prefix, "vtk");
    }
  }
}

// ---- Global ID management (parallel-friendly but not MPI-dependent)
index_t MeshBase::global_to_local_cell(gid_t gid) const {
  auto it = global2local_cell_.find(gid);
  return (it != global2local_cell_.end()) ? it->second : -1;
}

index_t MeshBase::global_to_local_node(gid_t gid) const {
  auto it = global2local_node_.find(gid);
  return (it != global2local_node_.end()) ? it->second : -1;
}

index_t MeshBase::global_to_local_face(gid_t gid) const {
  auto it = global2local_face_.find(gid);
  return (it != global2local_face_.end()) ? it->second : -1;
}

// ---- Adjacency queries
void MeshBase::build_node2cell() const {
  if (!node2cell_offsets_.empty()) return; // Already built

  size_t n_nodes = this->n_nodes();
  size_t n_cells = this->n_cells();

  // Count cells per node
  std::vector<index_t> counts(n_nodes, 0);
  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_node] = cell_nodes_span(static_cast<index_t>(c));
    for (size_t i = 0; i < n_node; ++i) {
      counts[nodes_ptr[i]]++;
    }
  }

  // Build offsets
  node2cell_offsets_.resize(n_nodes + 1);
  node2cell_offsets_[0] = 0;
  for (size_t n = 0; n < n_nodes; ++n) {
    node2cell_offsets_[n + 1] = node2cell_offsets_[n] + counts[n];
  }

  // Fill connectivity
  node2cell_.resize(static_cast<size_t>(node2cell_offsets_.back()));
  std::vector<offset_t> pos(n_nodes, 0);
  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_node] = cell_nodes_span(static_cast<index_t>(c));
    for (size_t i = 0; i < n_node; ++i) {
      index_t node_id = nodes_ptr[i];
      offset_t offset = node2cell_offsets_[node_id];
      node2cell_[static_cast<size_t>(offset + pos[node_id]++)] = static_cast<index_t>(c);
    }
  }
}

void MeshBase::build_node2face() const {
  if (!node2face_offsets_.empty()) return; // Already built

  size_t n_nodes = this->n_nodes();
  size_t n_faces = this->n_faces();

  if (n_faces == 0) return;

  // Count faces per node
  std::vector<index_t> counts(n_nodes, 0);
  for (size_t f = 0; f < n_faces; ++f) {
    auto [nodes_ptr, n_node] = face_nodes_span(static_cast<index_t>(f));
    for (size_t i = 0; i < n_node; ++i) {
      counts[nodes_ptr[i]]++;
    }
  }

  // Build offsets
  node2face_offsets_.resize(n_nodes + 1);
  node2face_offsets_[0] = 0;
  for (size_t n = 0; n < n_nodes; ++n) {
    node2face_offsets_[n + 1] = node2face_offsets_[n] + counts[n];
  }

  // Fill connectivity
  node2face_.resize(node2face_offsets_.back());
  std::vector<index_t> pos(n_nodes, 0);
  for (size_t f = 0; f < n_faces; ++f) {
    auto [nodes_ptr, n_node] = face_nodes_span(static_cast<index_t>(f));
    for (size_t i = 0; i < n_node; ++i) {
      index_t node_id = nodes_ptr[i];
      offset_t offset = node2face_offsets_[node_id];
      node2face_[offset + pos[node_id]++] = static_cast<index_t>(f);
    }
  }
}

void MeshBase::build_cell2cell() const {
  if (!cell2cell_offsets_.empty()) return; // Already built

  size_t n_cells = this->n_cells();

  // Build via faces if available
  if (n_faces() > 0) {
    std::vector<std::unordered_set<index_t>> neighbors(n_cells);

    for (size_t f = 0; f < n_faces(); ++f) {
      const auto& fc = face2cell_[f];
      if (fc[0] >= 0 && fc[1] >= 0) {
        neighbors[fc[0]].insert(fc[1]);
        neighbors[fc[1]].insert(fc[0]);
      }
    }

    // Convert to CSR
    cell2cell_offsets_.resize(n_cells + 1);
    cell2cell_offsets_[0] = 0;
    for (size_t c = 0; c < n_cells; ++c) {
      cell2cell_offsets_[c + 1] = cell2cell_offsets_[c] + static_cast<index_t>(neighbors[c].size());
    }

    cell2cell_.resize(cell2cell_offsets_.back());
    for (size_t c = 0; c < n_cells; ++c) {
      offset_t offset = cell2cell_offsets_[c];
      index_t pos = 0;
      for (index_t neighbor : neighbors[c]) {
        cell2cell_[offset + pos++] = neighbor;
      }
    }
  } else {
    // Build via shared nodes (slower)
    build_node2cell();

    std::vector<std::unordered_set<index_t>> neighbors(n_cells);

    for (size_t c = 0; c < n_cells; ++c) {
      auto [nodes_ptr, n_node] = cell_nodes_span(static_cast<index_t>(c));
      for (size_t i = 0; i < n_node; ++i) {
        index_t node_id = nodes_ptr[i];
        offset_t start = node2cell_offsets_[node_id];
        offset_t end = node2cell_offsets_[node_id + 1];
        for (offset_t j = start; j < end; ++j) {
          index_t other_cell = node2cell_[j];
          if (other_cell != static_cast<index_t>(c)) {
            neighbors[c].insert(other_cell);
          }
        }
      }
    }

    // Convert to CSR
    cell2cell_offsets_.resize(n_cells + 1);
    cell2cell_offsets_[0] = 0;
    for (size_t c = 0; c < n_cells; ++c) {
      cell2cell_offsets_[c + 1] = cell2cell_offsets_[c] + static_cast<index_t>(neighbors[c].size());
    }

    cell2cell_.resize(cell2cell_offsets_.back());
    for (size_t c = 0; c < n_cells; ++c) {
      offset_t offset = cell2cell_offsets_[c];
      index_t pos = 0;
      for (index_t neighbor : neighbors[c]) {
        cell2cell_[offset + pos++] = neighbor;
      }
    }
  }
}

std::vector<index_t> MeshBase::cell_neighbors(index_t c) const {
  build_cell2cell();

  if (c < 0 || static_cast<size_t>(c) >= cell2cell_offsets_.size() - 1) {
    return {};
  }

  offset_t start = cell2cell_offsets_[c];
  offset_t end = cell2cell_offsets_[c + 1];

  return std::vector<index_t>(cell2cell_.begin() + start, cell2cell_.begin() + end);
}

std::vector<index_t> MeshBase::node_cells(index_t n) const {
  build_node2cell();

  if (n < 0 || static_cast<size_t>(n) >= node2cell_offsets_.size() - 1) {
    return {};
  }

  offset_t start = node2cell_offsets_[n];
  offset_t end = node2cell_offsets_[n + 1];

  return std::vector<index_t>(node2cell_.begin() + start, node2cell_.begin() + end);
}

std::vector<index_t> MeshBase::face_neighbors(index_t f) const {
  if (f < 0 || static_cast<size_t>(f) >= face2cell_.size()) {
    return {};
  }

  std::vector<index_t> neighbors;
  const auto& fc = face2cell_[f];
  if (fc[0] >= 0) neighbors.push_back(fc[0]);
  if (fc[1] >= 0) neighbors.push_back(fc[1]);

  return neighbors;
}

std::vector<index_t> MeshBase::boundary_faces() const {
  std::vector<index_t> result;

  for (size_t f = 0; f < face2cell_.size(); ++f) {
    if (face2cell_[f][1] < 0) { // Outer cell is -1 for boundary
      result.push_back(static_cast<index_t>(f));
    }
  }

  return result;
}

std::vector<index_t> MeshBase::boundary_cells() const {
  std::unordered_set<index_t> boundary_set;

  for (size_t f = 0; f < face2cell_.size(); ++f) {
    if (face2cell_[f][1] < 0 && face2cell_[f][0] >= 0) {
      boundary_set.insert(face2cell_[f][0]);
    }
  }

  return std::vector<index_t>(boundary_set.begin(), boundary_set.end());
}

// ---- Submesh extraction
MeshBase MeshBase::extract_submesh_by_region(label_t region_label) const {
  return extract_submesh_by_regions({region_label});
}

MeshBase MeshBase::extract_submesh_by_regions(const std::vector<label_t>& region_labels) const {
  // Create set for quick lookup
  std::unordered_set<label_t> label_set(region_labels.begin(), region_labels.end());

  // Find cells to extract
  std::vector<index_t> cells_to_extract;
  for (size_t c = 0; c < n_cells(); ++c) {
    if (label_set.count(cell_region_id_[c])) {
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
  std::vector<index_t> new_cell2node_offsets = {0};
  std::vector<index_t> new_cell2node;

  for (index_t c : cells_to_extract) {
    new_cell_shapes.push_back(cell_shape_[c]);

    auto [nodes_ptr, n_node] = cell_nodes_span(c);
    for (size_t i = 0; i < n_node; ++i) {
      new_cell2node.push_back(old_to_new_node[nodes_ptr[i]]);
    }
    new_cell2node_offsets.push_back(static_cast<index_t>(new_cell2node.size()));
  }

  submesh.build_from_arrays(spatial_dim_, new_coords, new_cell2node_offsets,
                           new_cell2node, new_cell_shapes);

  return submesh;
}

MeshBase MeshBase::extract_submesh_by_boundary(label_t boundary_label) const {
  // Find faces with the boundary label
  auto boundary_face_ids = faces_with_label(boundary_label);

  // Find cells adjacent to these faces
  std::unordered_set<index_t> cell_set;
  for (index_t f : boundary_face_ids) {
    if (face2cell_[f][0] >= 0) {
      cell_set.insert(face2cell_[f][0]);
    }
  }

  // Convert to vector
  std::vector<index_t> cells_to_extract(cell_set.begin(), cell_set.end());
  std::sort(cells_to_extract.begin(), cells_to_extract.end());

  // Extract unique nodes from these cells
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

  // Copy region labels
  for (size_t i = 0; i < cells_to_extract.size(); ++i) {
    submesh.cell_region_id_[i] = cell_region_id_[cells_to_extract[i]];
  }

  return submesh;
}

// ---- Geometry queries
std::array<real_t,3> MeshBase::face_normal(index_t f, Configuration cfg) const {
  auto [nodes_ptr, n_nodes] = face_nodes_span(f);

  const std::vector<real_t>& coords = (cfg == Configuration::Current && !X_cur_.empty())
                                      ? X_cur_ : X_ref_;

  std::array<real_t,3> normal = {{0,0,0}};

  if (spatial_dim_ == 2) {
    // 2D: normal to line
    if (n_nodes >= 2) {
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};

      for (int d = 0; d < spatial_dim_; ++d) {
        p0[d] = coords[nodes_ptr[0] * spatial_dim_ + d];
        p1[d] = coords[nodes_ptr[1] * spatial_dim_ + d];
      }

      // 90-degree rotation in 2D
      normal[0] = -(p1[1] - p0[1]);
      normal[1] = p1[0] - p0[0];
    }
  } else if (spatial_dim_ == 3) {
    // 3D: use cross product of two edges
    if (n_nodes >= 3) {
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      std::array<real_t,3> p2 = {{0,0,0}};

      for (int d = 0; d < spatial_dim_; ++d) {
        p0[d] = coords[nodes_ptr[0] * spatial_dim_ + d];
        p1[d] = coords[nodes_ptr[1] * spatial_dim_ + d];
        p2[d] = coords[nodes_ptr[2] * spatial_dim_ + d];
      }

      // Edges
      std::array<real_t,3> e1, e2;
      for (int d = 0; d < 3; ++d) {
        e1[d] = p1[d] - p0[d];
        e2[d] = p2[d] - p0[d];
      }

      // Cross product
      normal[0] = e1[1] * e2[2] - e1[2] * e2[1];
      normal[1] = e1[2] * e2[0] - e1[0] * e2[2];
      normal[2] = e1[0] * e2[1] - e1[1] * e2[0];
    }
  }

  // Normalize
  real_t norm = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
  if (norm > 1e-12) {
    for (int d = 0; d < 3; ++d) {
      normal[d] /= norm;
    }
  }

  return normal;
}

real_t MeshBase::face_area(index_t f, Configuration cfg) const {
  auto [nodes_ptr, n_nodes] = face_nodes_span(f);

  const std::vector<real_t>& coords = (cfg == Configuration::Current && !X_cur_.empty())
                                      ? X_cur_ : X_ref_;

  if (spatial_dim_ == 2) {
    // 2D: length of line
    if (n_nodes >= 2) {
      real_t len_sq = 0;
      for (int d = 0; d < spatial_dim_; ++d) {
        real_t diff = coords[nodes_ptr[1] * spatial_dim_ + d]
                    - coords[nodes_ptr[0] * spatial_dim_ + d];
        len_sq += diff * diff;
      }
      return std::sqrt(len_sq);
    }
  } else if (spatial_dim_ == 3) {
    // 3D: area of polygon (simplified for triangle)
    if (n_nodes == 3) {
      // Triangle area via cross product
      auto normal = face_normal(f, cfg);
      real_t area = 0;

      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      std::array<real_t,3> p2 = {{0,0,0}};

      for (int d = 0; d < spatial_dim_; ++d) {
        p0[d] = coords[nodes_ptr[0] * spatial_dim_ + d];
        p1[d] = coords[nodes_ptr[1] * spatial_dim_ + d];
        p2[d] = coords[nodes_ptr[2] * spatial_dim_ + d];
      }

      std::array<real_t,3> e1, e2;
      for (int d = 0; d < 3; ++d) {
        e1[d] = p1[d] - p0[d];
        e2[d] = p2[d] - p0[d];
      }

      std::array<real_t,3> cross;
      cross[0] = e1[1] * e2[2] - e1[2] * e2[1];
      cross[1] = e1[2] * e2[0] - e1[0] * e2[2];
      cross[2] = e1[0] * e2[1] - e1[1] * e2[0];

      area = 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
      return area;
    } else if (n_nodes >= 4) {
      // General polygon area using shoelace formula
      // First project polygon to best-fit plane
      auto normal = face_normal(f, cfg);

      // Find dominant axis of normal (for projection)
      int dom_axis = 0;
      real_t max_comp = std::abs(normal[0]);
      if (std::abs(normal[1]) > max_comp) {
        dom_axis = 1;
        max_comp = std::abs(normal[1]);
      }
      if (std::abs(normal[2]) > max_comp) {
        dom_axis = 2;
      }

      // Project to 2D (remove dominant axis)
      int u_axis = (dom_axis + 1) % 3;
      int v_axis = (dom_axis + 2) % 3;

      // Apply shoelace formula
      real_t area = 0;
      for (size_t i = 0; i < n_nodes; ++i) {
        size_t j = (i + 1) % n_nodes;
        real_t u_i = coords[nodes_ptr[i] * spatial_dim_ + u_axis];
        real_t v_i = coords[nodes_ptr[i] * spatial_dim_ + v_axis];
        real_t u_j = coords[nodes_ptr[j] * spatial_dim_ + u_axis];
        real_t v_j = coords[nodes_ptr[j] * spatial_dim_ + v_axis];
        area += u_i * v_j - u_j * v_i;
      }
      area = std::abs(area) * 0.5;

      // Correct for projection
      area /= std::abs(normal[dom_axis]);

      return area;
    }
  }

  return 0.0;
}

real_t MeshBase::cell_measure(index_t c, Configuration cfg) const {
  auto [nodes_ptr, n_nodes] = cell_nodes_span(c);

  const std::vector<real_t>& coords = (cfg == Configuration::Current && !X_cur_.empty())
                                      ? X_cur_ : X_ref_;

  const CellShape& shape = cell_shape_[c];

  // Helper lambda to get node coordinates
  auto get_coords = [&](index_t node_id) -> std::array<double,3> {
    std::array<double,3> pt = {{0,0,0}};
    for (int d = 0; d < spatial_dim_; ++d) {
      pt[d] = coords[node_id * spatial_dim_ + d];
    }
    return pt;
  };

  // Based on cell family
  switch (shape.family) {
    case CellFamily::Line: {
      // Length of line segment
      if (n_nodes >= 2) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        double len_sq = 0;
        for (int d = 0; d < spatial_dim_; ++d) {
          double diff = p1[d] - p0[d];
          len_sq += diff * diff;
        }
        return std::sqrt(len_sq);
      }
      break;
    }

    case CellFamily::Triangle: {
      // Triangle area via cross product
      if (n_nodes >= 3) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);

        // Vectors from p0 to p1 and p2
        std::array<real_t,3> e1, e2;
        for (int d = 0; d < 3; ++d) {
          e1[d] = p1[d] - p0[d];
          e2[d] = p2[d] - p0[d];
        }

        if (spatial_dim_ == 2) {
          // 2D: det(e1, e2) / 2
          return 0.5 * std::abs(e1[0] * e2[1] - e1[1] * e2[0]);
        } else {
          // 3D: |e1 x e2| / 2
          std::array<real_t,3> cross;
          cross[0] = e1[1] * e2[2] - e1[2] * e2[1];
          cross[1] = e1[2] * e2[0] - e1[0] * e2[2];
          cross[2] = e1[0] * e2[1] - e1[1] * e2[0];
          return 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
        }
      }
      break;
    }

    case CellFamily::Quad: {
      // Quad area - split into two triangles
      if (n_nodes >= 4) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);
        auto p3 = get_coords(nodes_ptr[3]);

        // Compute as average of two diagonal splits
        real_t area1 = 0, area2 = 0;

        if (spatial_dim_ == 2) {
          // Triangle 0-1-2
          real_t a1 = 0.5 * std::abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1]) -
            (p2[0] - p0[0]) * (p1[1] - p0[1])
          );
          // Triangle 0-2-3
          real_t a2 = 0.5 * std::abs(
            (p2[0] - p0[0]) * (p3[1] - p0[1]) -
            (p3[0] - p0[0]) * (p2[1] - p0[1])
          );
          return a1 + a2;
        } else {
          // 3D quad - use cross products
          std::array<real_t,3> e1, e2, e3;
          for (int d = 0; d < 3; ++d) {
            e1[d] = p1[d] - p0[d];
            e2[d] = p2[d] - p0[d];
            e3[d] = p3[d] - p0[d];
          }

          // Area of triangle 0-1-2
          std::array<real_t,3> cross1;
          cross1[0] = e1[1] * e2[2] - e1[2] * e2[1];
          cross1[1] = e1[2] * e2[0] - e1[0] * e2[2];
          cross1[2] = e1[0] * e2[1] - e1[1] * e2[0];
          area1 = 0.5 * std::sqrt(cross1[0]*cross1[0] + cross1[1]*cross1[1] + cross1[2]*cross1[2]);

          // Area of triangle 0-2-3
          std::array<real_t,3> cross2;
          cross2[0] = e2[1] * e3[2] - e2[2] * e3[1];
          cross2[1] = e2[2] * e3[0] - e2[0] * e3[2];
          cross2[2] = e2[0] * e3[1] - e2[1] * e3[0];
          area2 = 0.5 * std::sqrt(cross2[0]*cross2[0] + cross2[1]*cross2[1] + cross2[2]*cross2[2]);

          return area1 + area2;
        }
      }
      break;
    }

    case CellFamily::Polygon: {
      // General polygon - simplified triangulation from first vertex
      if (n_nodes >= 3) {
        real_t total_area = 0;
        auto p0 = get_coords(nodes_ptr[0]);

        for (size_t i = 1; i < n_nodes - 1; ++i) {
          auto p1 = get_coords(nodes_ptr[i]);
          auto p2 = get_coords(nodes_ptr[i + 1]);

          std::array<real_t,3> e1, e2;
          for (int d = 0; d < 3; ++d) {
            e1[d] = p1[d] - p0[d];
            e2[d] = p2[d] - p0[d];
          }

          if (spatial_dim_ == 2) {
            total_area += 0.5 * std::abs(e1[0] * e2[1] - e1[1] * e2[0]);
          } else {
            std::array<real_t,3> cross;
            cross[0] = e1[1] * e2[2] - e1[2] * e2[1];
            cross[1] = e1[2] * e2[0] - e1[0] * e2[2];
            cross[2] = e1[0] * e2[1] - e1[1] * e2[0];
            total_area += 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
          }
        }
        return total_area;
      }
      break;
    }

    case CellFamily::Tetra: {
      // Tetrahedron volume via scalar triple product
      if (n_nodes >= 4) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);
        auto p3 = get_coords(nodes_ptr[3]);

        // Vectors from p0
        std::array<real_t,3> v1, v2, v3;
        for (int d = 0; d < 3; ++d) {
          v1[d] = p1[d] - p0[d];
          v2[d] = p2[d] - p0[d];
          v3[d] = p3[d] - p0[d];
        }

        // Scalar triple product: v1 . (v2 x v3)
        real_t det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                   - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                   + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);

        return std::abs(det) / 6.0;
      }
      break;
    }

    case CellFamily::Hex: {
      // Hexahedron volume - decompose into 5 tetrahedra
      if (n_nodes >= 8) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);
        auto p3 = get_coords(nodes_ptr[3]);
        auto p4 = get_coords(nodes_ptr[4]);
        auto p5 = get_coords(nodes_ptr[5]);
        auto p6 = get_coords(nodes_ptr[6]);
        auto p7 = get_coords(nodes_ptr[7]);

        // Helper for tet volume
        auto tet_vol = [](const std::array<real_t,3>& a, const std::array<real_t,3>& b,
                         const std::array<real_t,3>& c, const std::array<real_t,3>& d) -> real_t {
          std::array<real_t,3> v1, v2, v3;
          for (int i = 0; i < 3; ++i) {
            v1[i] = b[i] - a[i];
            v2[i] = c[i] - a[i];
            v3[i] = d[i] - a[i];
          }
          real_t det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                     - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                     + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
          return det / 6.0;
        };

        // Decomposition into 5 tets
        real_t vol = 0;
        vol += tet_vol(p0, p1, p2, p5);
        vol += tet_vol(p0, p2, p7, p5);
        vol += tet_vol(p0, p2, p3, p7);
        vol += tet_vol(p0, p5, p7, p4);
        vol += tet_vol(p2, p7, p5, p6);

        return std::abs(vol);
      }
      break;
    }

    case CellFamily::Wedge: {
      // Wedge (prism) volume - decompose into 3 tetrahedra
      if (n_nodes >= 6) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);
        auto p3 = get_coords(nodes_ptr[3]);
        auto p4 = get_coords(nodes_ptr[4]);
        auto p5 = get_coords(nodes_ptr[5]);

        // Helper for tet volume
        auto tet_vol = [](const std::array<real_t,3>& a, const std::array<real_t,3>& b,
                         const std::array<real_t,3>& c, const std::array<real_t,3>& d) -> real_t {
          std::array<real_t,3> v1, v2, v3;
          for (int i = 0; i < 3; ++i) {
            v1[i] = b[i] - a[i];
            v2[i] = c[i] - a[i];
            v3[i] = d[i] - a[i];
          }
          real_t det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                     - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                     + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
          return det / 6.0;
        };

        // Decomposition into 3 tets
        real_t vol = 0;
        vol += tet_vol(p0, p1, p2, p5);
        vol += tet_vol(p0, p1, p5, p4);
        vol += tet_vol(p0, p4, p5, p3);

        return std::abs(vol);
      }
      break;
    }

    case CellFamily::Pyramid: {
      // Pyramid volume = (1/3) * base_area * height
      if (n_nodes >= 5) {
        // Base is quad 0-1-2-3, apex is 4
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);
        auto p3 = get_coords(nodes_ptr[3]);
        auto p4 = get_coords(nodes_ptr[4]);

        // Split base into two triangles and compute volumes
        auto tet_vol = [](const std::array<real_t,3>& a, const std::array<real_t,3>& b,
                         const std::array<real_t,3>& c, const std::array<real_t,3>& d) -> real_t {
          std::array<real_t,3> v1, v2, v3;
          for (int i = 0; i < 3; ++i) {
            v1[i] = b[i] - a[i];
            v2[i] = c[i] - a[i];
            v3[i] = d[i] - a[i];
          }
          real_t det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                     - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                     + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
          return det / 6.0;
        };

        real_t vol = 0;
        vol += tet_vol(p0, p1, p2, p4);
        vol += tet_vol(p0, p2, p3, p4);

        return std::abs(vol);
      }
      break;
    }

    case CellFamily::Polyhedron: {
      // General polyhedron - would need face information
      // Placeholder for now
      break;
    }

    default:
      break;
  }

  return 0.0;
}

// Helper for tetrahedron volume
real_t MeshBase::tet_volume(const std::array<real_t,3>& p0,
                            const std::array<real_t,3>& p1,
                            const std::array<real_t,3>& p2,
                            const std::array<real_t,3>& p3) const {
  // Vectors from p0
  std::array<real_t,3> v1, v2, v3;
  for (int d = 0; d < 3; ++d) {
    v1[d] = p1[d] - p0[d];
    v2[d] = p2[d] - p0[d];
    v3[d] = p3[d] - p0[d];
  }

  // Scalar triple product
  real_t det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
             - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
             + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);

  return std::abs(det) / 6.0;
}

void MeshBase::use_reference_configuration() {
  active_config_ = Configuration::Reference;
  invalidate_caches();
}

void MeshBase::use_current_configuration() {
  if (X_cur_.empty()) {
    throw std::runtime_error("No current configuration available");
  }
  active_config_ = Configuration::Current;
  invalidate_caches();
}

void MeshBase::set_displacement_field(const FieldHandle& U) {
  // Would compute X_cur = X_ref + U
  // Placeholder for now
  invalidate_caches();
}

// ---- Quality metrics
real_t MeshBase::compute_quality(index_t cell, const std::string& metric) const {
  // Placeholder for quality metrics
  // Would implement aspect ratio, skewness, etc. based on cell shape

  if (metric == "aspect_ratio") {
    // Simple estimate from bounding box
    auto [nodes_ptr, n_nodes] = cell_nodes_span(cell);

    std::array<real_t,3> min_pt = {{1e300, 1e300, 1e300}};
    std::array<real_t,3> max_pt = {{-1e300, -1e300, -1e300}};

    for (size_t i = 0; i < n_nodes; ++i) {
      index_t node_id = nodes_ptr[i];
      for (int d = 0; d < spatial_dim_; ++d) {
        real_t val = X_ref_[node_id * spatial_dim_ + d];
        min_pt[d] = std::min(min_pt[d], val);
        max_pt[d] = std::max(max_pt[d], val);
      }
    }

    real_t min_len = 1e300;
    real_t max_len = -1e300;

    for (int d = 0; d < spatial_dim_; ++d) {
      real_t len = max_pt[d] - min_pt[d];
      if (len > 1e-12) {
        min_len = std::min(min_len, len);
        max_len = std::max(max_len, len);
      }
    }

    return (min_len > 1e-12) ? max_len / min_len : 1e300;
  }

  return 1.0;
}

std::pair<real_t,real_t> MeshBase::global_quality_range(const std::string& metric) const {
  real_t min_quality = 1e300;
  real_t max_quality = -1e300;

  for (size_t c = 0; c < n_cells(); ++c) {
    real_t q = compute_quality(static_cast<index_t>(c), metric);
    min_quality = std::min(min_quality, q);
    max_quality = std::max(max_quality, q);
  }

  return {min_quality, max_quality};
}

// ---- Search structures & point location
void MeshBase::build_search_structure(Configuration cfg) const {
  if (!search_accel_) {
    search_accel_ = std::make_unique<SearchAccel>();
  }
  search_accel_->build(*this, cfg);
}

void MeshBase::clear_search_structure() const {
  search_accel_.reset();
}

MeshBase::PointLocateResult MeshBase::locate_point(const std::array<real_t,3>& x,
                                                   Configuration cfg) const {
  // Build search structure if needed, or rebuild if built for wrong configuration
  if (!search_accel_ || search_accel_->built_for_config != cfg) {
    const_cast<MeshBase*>(this)->build_search_structure(cfg);
  }

  PointLocateResult result;

  // First check global bounding box
  if (!search_accel_->global_box.contains(x)) {
    return result; // Not found
  }

  // Check each cell (brute force for now)
  for (size_t c = 0; c < n_cells(); ++c) {
    if (search_accel_->cell_boxes[c].contains(x)) {
      // Would do proper point-in-cell test here
      // For now, just check if in bounding box
      result.cell_id = static_cast<index_t>(c);
      result.found = true;
      // Would compute actual parametric coordinates xi
      result.xi = {{0.5, 0.5, 0.5}}; // Placeholder
      break;
    }
  }

  return result;
}

std::vector<MeshBase::PointLocateResult> MeshBase::locate_points(
    const std::vector<std::array<real_t,3>>& X, Configuration cfg) const {

  std::vector<PointLocateResult> results;
  results.reserve(X.size());

  for (const auto& x : X) {
    results.push_back(locate_point(x, cfg));
  }

  return results;
}

MeshBase::RayIntersectResult MeshBase::intersect_ray(const std::array<real_t,3>& origin,
                                                     const std::array<real_t,3>& direction,
                                                     Configuration cfg) const {
  RayIntersectResult result;

  // Placeholder - would implement ray-face intersection
  // For boundary faces, check ray-triangle/quad intersection

  return result;
}

// ---- Adaptivity methods
void MeshBase::mark_refine(const std::vector<index_t>& cells) {
  if (refine_flag_.size() < n_cells()) {
    refine_flag_.resize(n_cells(), 0);
  }

  for (index_t c : cells) {
    if (c >= 0 && static_cast<size_t>(c) < refine_flag_.size()) {
      refine_flag_[c] = 1;
    }
  }
}

void MeshBase::mark_coarsen(const std::vector<index_t>& cells) {
  if (coarsen_flag_.size() < n_cells()) {
    coarsen_flag_.resize(n_cells(), 0);
  }

  for (index_t c : cells) {
    if (c >= 0 && static_cast<size_t>(c) < coarsen_flag_.size()) {
      coarsen_flag_[c] = 1;
    }
  }
}

void MeshBase::apply_refinement() {
  // Placeholder for actual refinement
  // Would subdivide marked cells
}

void MeshBase::apply_coarsening() {
  // Placeholder for actual coarsening
  // Would merge marked cells
}

index_t MeshBase::parent(index_t cell) const {
  if (cell < 0 || static_cast<size_t>(cell) >= cell_parent_.size()) {
    return -1;
  }
  return cell_parent_[cell];
}

std::vector<index_t> MeshBase::children(index_t cell) const {
  if (cell < 0 || static_cast<size_t>(cell) >= cell_children_offsets_.size() - 1) {
    return {};
  }

  offset_t start = cell_children_offsets_[cell];
  offset_t end = cell_children_offsets_[cell + 1];

  return std::vector<index_t>(cell_children_.begin() + start, cell_children_.begin() + end);
}

// ---- Periodic/mortar/contact constraints
void MeshBase::register_periodic_pair(const std::string& face_set_A, const std::string& face_set_B,
                                      const std::array<real_t,9>& transform) {
  periodic_pairs_.push_back({face_set_A, face_set_B, transform});
}

void MeshBase::register_mortar_interface(const std::string& master_set, const std::string& slave_set) {
  mortar_interfaces_.push_back({master_set, slave_set});
}

void MeshBase::register_contact_candidates(const std::string& setA, const std::string& setB) {
  contact_candidates_.push_back({setA, setB});
}

// ---- Memory management
void MeshBase::shrink_to_fit() {
  X_ref_.shrink_to_fit();
  X_cur_.shrink_to_fit();

  cell_shape_.shrink_to_fit();
  cell2node_offsets_.shrink_to_fit();
  cell2node_.shrink_to_fit();

  face_shape_.shrink_to_fit();
  face2node_offsets_.shrink_to_fit();
  face2node_.shrink_to_fit();
  face2cell_.shrink_to_fit();

  // ... shrink all other vectors
}

size_t MeshBase::memory_usage_bytes() const {
  size_t bytes = 0;

  bytes += X_ref_.capacity() * sizeof(real_t);
  bytes += X_cur_.capacity() * sizeof(real_t);

  bytes += cell_shape_.capacity() * sizeof(CellShape);
  bytes += cell2node_offsets_.capacity() * sizeof(index_t);
  bytes += cell2node_.capacity() * sizeof(index_t);

  bytes += face_shape_.capacity() * sizeof(CellShape);
  bytes += face2node_offsets_.capacity() * sizeof(index_t);
  bytes += face2node_.capacity() * sizeof(index_t);
  bytes += face2cell_.capacity() * sizeof(std::array<index_t,2>);

  // ... add other data structures

  return bytes;
}

// ---- Builder utilities
MeshBase MeshBase::build_cartesian(int nx, int ny, int nz, const BoundingBox& domain) {
  int dim = (nz > 1) ? 3 : (ny > 1) ? 2 : 1;
  MeshBase mesh(dim);

  // Generate regular grid nodes
  std::vector<real_t> coords;
  int n_nodes = (nx + 1) * (ny + 1) * (nz + 1);
  coords.reserve(n_nodes * dim);

  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        real_t x = domain.min[0] + i * (domain.max[0] - domain.min[0]) / nx;
        coords.push_back(x);

        if (dim > 1) {
          real_t y = domain.min[1] + j * (domain.max[1] - domain.min[1]) / ny;
          coords.push_back(y);
        }

        if (dim > 2) {
          real_t z = domain.min[2] + k * (domain.max[2] - domain.min[2]) / nz;
          coords.push_back(z);
        }
      }
    }
  }

  // Generate cells
  std::vector<CellShape> cell_shapes;
  std::vector<index_t> cell2node_offsets = {0};
  std::vector<index_t> cell2node;

  if (dim == 1) {
    // Lines
    CellShape line_shape = {3, CellFamily::Line, 2, 1, false};
    for (int i = 0; i < nx; ++i) {
      cell_shapes.push_back(line_shape);
      cell2node.push_back(i);
      cell2node.push_back(i + 1);
      cell2node_offsets.push_back(static_cast<index_t>(cell2node.size()));
    }
  } else if (dim == 2) {
    // Quads
    CellShape quad_shape = {9, CellFamily::Quad, 4, 1, false};
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        cell_shapes.push_back(quad_shape);

        int n00 = j * (nx + 1) + i;
        int n10 = n00 + 1;
        int n01 = n00 + (nx + 1);
        int n11 = n01 + 1;

        cell2node.push_back(n00);
        cell2node.push_back(n10);
        cell2node.push_back(n11);
        cell2node.push_back(n01);

        cell2node_offsets.push_back(static_cast<index_t>(cell2node.size()));
      }
    }
  } else {
    // Hexes
    CellShape hex_shape = {12, CellFamily::Hex, 8, 1, false};
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          cell_shapes.push_back(hex_shape);

          int n000 = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
          int n100 = n000 + 1;
          int n010 = n000 + (nx + 1);
          int n110 = n010 + 1;
          int n001 = n000 + (ny + 1) * (nx + 1);
          int n101 = n001 + 1;
          int n011 = n001 + (nx + 1);
          int n111 = n011 + 1;

          cell2node.push_back(n000);
          cell2node.push_back(n100);
          cell2node.push_back(n110);
          cell2node.push_back(n010);
          cell2node.push_back(n001);
          cell2node.push_back(n101);
          cell2node.push_back(n111);
          cell2node.push_back(n011);

          cell2node_offsets.push_back(static_cast<index_t>(cell2node.size()));
        }
      }
    }
  }

  mesh.build_from_arrays(dim, coords, cell2node_offsets, cell2node, cell_shapes);

  return mesh;
}

MeshBase MeshBase::build_extruded(const MeshBase& base_2d, int n_layers, real_t height) {
  if (base_2d.dim() != 2) {
    throw std::invalid_argument("Base mesh must be 2D");
  }

  // Placeholder - would extrude 2D mesh to 3D
  return MeshBase();
}

// ---- Helpers
size_t MeshBase::entity_count(EntityKind k) const noexcept {
  switch (k) {
    case EntityKind::Vertex: return n_nodes();
    case EntityKind::Edge:   return n_edges();
    case EntityKind::Face:   return n_faces();
    case EntityKind::Cell:   return n_cells();
  }
  return 0;
}

void MeshBase::invalidate_caches() {
  geom_cache_valid_ = false;
  cell_measures_.clear();
  face_areas_.clear();
  face_normals_.clear();

  // Clear adjacency caches
  node2cell_offsets_.clear();
  node2cell_.clear();
  node2face_offsets_.clear();
  node2face_.clear();
  cell2cell_offsets_.clear();
  cell2cell_.clear();

  // Clear search structure
  search_accel_.reset();
}

// ---- IO registry
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
  auto& readers = readers_();
  auto it = readers.find(opts.format);
  if (it == readers.end()) {
    throw std::runtime_error("Unknown mesh format: " + opts.format);
  }
  return it->second(opts);
}

void MeshBase::save(const MeshIOOptions& opts) const {
  auto& writers = writers_();
  auto it = writers.find(opts.format);
  if (it == writers.end()) {
    throw std::runtime_error("Unknown mesh format: " + opts.format);
  }
  it->second(*this, opts);
}

std::vector<std::string> MeshBase::registered_readers() {
  std::vector<std::string> formats;
  for (const auto& [format, fn] : readers_()) {
    formats.push_back(format);
  }
  return formats;
}

std::vector<std::string> MeshBase::registered_writers() {
  std::vector<std::string> formats;
  for (const auto& [format, fn] : writers_()) {
    formats.push_back(format);
  }
  return formats;
}

} // namespace svmp