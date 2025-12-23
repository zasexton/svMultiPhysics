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

#include "MeshFields.h"
#include "../Core/MeshBase.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <cstring>

namespace svmp {

namespace {

template <typename T>
MeshFields::FieldStats compute_stats_impl(const T* data,
                                         size_t n_entities,
                                         size_t n_components,
                                         size_t component)
{
  MeshFields::FieldStats stats;
  if (n_entities == 0) {
    return stats;
  }

  const real_t first = static_cast<real_t>(data[component]);
  stats.min = first;
  stats.max = first;
  stats.sum = 0.0;

  for (size_t i = 0; i < n_entities; ++i) {
    const real_t val = static_cast<real_t>(data[i * n_components + component]);
    stats.min = std::min(stats.min, val);
    stats.max = std::max(stats.max, val);
    stats.sum += val;
  }

  stats.mean = stats.sum / static_cast<real_t>(n_entities);

  real_t sum_sq_diff = 0.0;
  for (size_t i = 0; i < n_entities; ++i) {
    const real_t val = static_cast<real_t>(data[i * n_components + component]);
    const real_t diff = val - stats.mean;
    sum_sq_diff += diff * diff;
  }

  stats.std_dev = std::sqrt(sum_sq_diff / static_cast<real_t>(n_entities));
  return stats;
}

template <typename T>
real_t compute_l2_norm_impl(const T* data, size_t n_total)
{
  long double sum_sq = 0.0L;
  for (size_t i = 0; i < n_total; ++i) {
    const long double v = static_cast<long double>(data[i]);
    sum_sq += v * v;
  }
  return std::sqrt(static_cast<real_t>(sum_sq));
}

template <typename T>
real_t compute_inf_norm_impl(const T* data, size_t n_total)
{
  long double max_val = 0.0L;
  for (size_t i = 0; i < n_total; ++i) {
    const long double v = std::abs(static_cast<long double>(data[i]));
    max_val = std::max(max_val, v);
  }
  return static_cast<real_t>(max_val);
}

} // namespace

namespace {

template <typename T>
void apply_injection_map(const T* src,
                         T* dst,
                         size_t components,
                         const EntityTransferMap& map)
{
  // Preconditions: map.validate(true) and map.is_injection() already checked by caller.
  for (size_t di = 0; di < map.dst_count; ++di) {
    const size_t k = static_cast<size_t>(map.dst_offsets[di]);
    const index_t si = map.src_indices[k];
    const size_t dst_off = di * components;
    const size_t src_off = static_cast<size_t>(si) * components;
    for (size_t c = 0; c < components; ++c) {
      dst[dst_off + c] = src[src_off + c];
    }
  }
}

template <typename SrcT, typename DstT, typename AccT>
void apply_weighted_sum_map(const SrcT* src,
                            DstT* dst,
                            size_t components,
                            const EntityTransferMap& map)
{
  for (size_t di = 0; di < map.dst_count; ++di) {
    const size_t begin = static_cast<size_t>(map.dst_offsets[di]);
    const size_t end = static_cast<size_t>(map.dst_offsets[di + 1]);

    for (size_t c = 0; c < components; ++c) {
      AccT sum = 0;
      for (size_t k = begin; k < end; ++k) {
        const index_t si = map.src_indices[k];
        const AccT w = static_cast<AccT>(map.weights[k]);
        sum += w * static_cast<AccT>(src[static_cast<size_t>(si) * components + c]);
      }
      dst[di * components + c] = static_cast<DstT>(sum);
    }
  }
}

void apply_injection_custom(const uint8_t* src,
                            uint8_t* dst,
                            size_t bytes_per_entity,
                            const EntityTransferMap& map)
{
  // Preconditions: map.validate(true) and map.is_injection() already checked by caller.
  for (size_t di = 0; di < map.dst_count; ++di) {
    const size_t k = static_cast<size_t>(map.dst_offsets[di]);
    const index_t si = map.src_indices[k];
    const uint8_t* src_ptr = src + static_cast<size_t>(si) * bytes_per_entity;
    uint8_t* dst_ptr = dst + di * bytes_per_entity;
    std::memcpy(dst_ptr, src_ptr, bytes_per_entity);
  }
}

void validate_transfer_common(const MeshBase& src_mesh,
                              const MeshBase& dst_mesh,
                              const FieldHandle& src_field,
                              const FieldHandle& dst_field,
                              const EntityTransferMap& map)
{
  if (src_field.kind != dst_field.kind) {
    throw std::invalid_argument("field transfer: field locations must match");
  }
  if (map.kind != src_field.kind) {
    throw std::invalid_argument("field transfer: map kind does not match field location");
  }

  const auto src_type = src_mesh.field_type(src_field);
  const auto dst_type = dst_mesh.field_type(dst_field);
  if (src_type != dst_type) {
    throw std::runtime_error("field transfer: scalar type mismatch");
  }

  const size_t src_components = src_mesh.field_components(src_field);
  const size_t dst_components = dst_mesh.field_components(dst_field);
  if (src_components != dst_components) {
    throw std::runtime_error("field transfer: component count mismatch");
  }

  const size_t src_count = src_mesh.field_entity_count(src_field);
  const size_t dst_count = dst_mesh.field_entity_count(dst_field);
  if (map.src_count != src_count) {
    throw std::invalid_argument("field transfer: map src_count mismatch");
  }
  if (map.dst_count != dst_count) {
    throw std::invalid_argument("field transfer: map dst_count mismatch");
  }

  map.validate(true);
}

void transfer_field_with_map(const MeshBase& src_mesh,
                             MeshBase& dst_mesh,
                             const FieldHandle& src_field,
                             const FieldHandle& dst_field,
                             const EntityTransferMap& map)
{
  validate_transfer_common(src_mesh, dst_mesh, src_field, dst_field, map);

  const void* src_raw = src_mesh.field_data(src_field);
  void* dst_raw = dst_mesh.field_data(dst_field);
  if (!src_raw || !dst_raw) {
    throw std::runtime_error("field transfer: invalid field handles");
  }

  const auto type = src_mesh.field_type(src_field);
  const size_t components = src_mesh.field_components(src_field);

  switch (type) {
    case FieldScalarType::Float64:
      apply_weighted_sum_map<double, double, long double>(
          static_cast<const double*>(src_raw), static_cast<double*>(dst_raw), components, map);
      break;
    case FieldScalarType::Float32:
      apply_weighted_sum_map<float, float, double>(
          static_cast<const float*>(src_raw), static_cast<float*>(dst_raw), components, map);
      break;

    case FieldScalarType::Int32: {
      if (!map.is_injection(0.0)) {
        throw std::invalid_argument("field transfer: Int32 fields require an injection map");
      }
      apply_injection_map<int32_t>(static_cast<const int32_t*>(src_raw),
                                   static_cast<int32_t*>(dst_raw),
                                   components,
                                   map);
      break;
    }
    case FieldScalarType::Int64: {
      if (!map.is_injection(0.0)) {
        throw std::invalid_argument("field transfer: Int64 fields require an injection map");
      }
      apply_injection_map<int64_t>(static_cast<const int64_t*>(src_raw),
                                   static_cast<int64_t*>(dst_raw),
                                   components,
                                   map);
      break;
    }
    case FieldScalarType::UInt8: {
      if (!map.is_injection(0.0)) {
        throw std::invalid_argument("field transfer: UInt8 fields require an injection map");
      }
      apply_injection_map<uint8_t>(static_cast<const uint8_t*>(src_raw),
                                   static_cast<uint8_t*>(dst_raw),
                                   components,
                                   map);
      break;
    }

    case FieldScalarType::Custom: {
      if (!map.is_injection(0.0)) {
        throw std::invalid_argument("field transfer: Custom fields require an injection map");
      }
      const size_t src_bpe = src_mesh.field_bytes_per_entity(src_field);
      const size_t dst_bpe = dst_mesh.field_bytes_per_entity(dst_field);
      if (src_bpe != dst_bpe) {
        throw std::runtime_error("field transfer: Custom field byte stride mismatch");
      }
      apply_injection_custom(static_cast<const uint8_t*>(src_raw),
                             static_cast<uint8_t*>(dst_raw),
                             src_bpe,
                             map);
      break;
    }
  }

  dst_mesh.event_bus().notify(MeshEvent::FieldsChanged);
}

} // namespace

// ---- Helper functions ----

size_t MeshFields::entity_count(const MeshBase& mesh, EntityKind kind) {
  switch (kind) {
    case EntityKind::Vertex: return mesh.n_vertices();
    case EntityKind::Edge: return mesh.n_edges();
    case EntityKind::Face: return mesh.n_faces();
    case EntityKind::Volume: return mesh.n_cells();
    default: return 0;
  }
}

// ---- Field attachment ----

FieldHandle MeshFields::attach_field(MeshBase& mesh,
                                    EntityKind kind,
                                    const std::string& name,
                                    FieldScalarType type,
                                    size_t components,
                                    size_t custom_bytes_per_component) {
  // Delegate to MeshBase implementation
  return mesh.attach_field(kind, name, type, components, custom_bytes_per_component);
}

FieldHandle MeshFields::attach_field_with_descriptor(MeshBase& mesh,
                                                    EntityKind kind,
                                                    const std::string& name,
                                                    FieldScalarType type,
                                                    const FieldDescriptor& descriptor) {
  // Delegate to MeshBase implementation
  return mesh.attach_field_with_descriptor(kind, name, type, descriptor);
}

void MeshFields::remove_field(MeshBase& mesh, const FieldHandle& handle) {
  mesh.remove_field(handle);
}

bool MeshFields::has_field(const MeshBase& mesh, EntityKind kind, const std::string& name) {
  return mesh.has_field(kind, name);
}

// ---- Field access ----

void* MeshFields::field_data(MeshBase& mesh, const FieldHandle& handle) {
  return mesh.field_data(handle);
}

const void* MeshFields::field_data(const MeshBase& mesh, const FieldHandle& handle) {
  return mesh.field_data(handle);
}

size_t MeshFields::field_components(const MeshBase& mesh, const FieldHandle& handle) {
  return mesh.field_components(handle);
}

FieldScalarType MeshFields::field_type(const MeshBase& mesh, const FieldHandle& handle) {
  return mesh.field_type(handle);
}

size_t MeshFields::field_entity_count(const MeshBase& mesh, const FieldHandle& handle) {
  return mesh.field_entity_count(handle);
}

size_t MeshFields::field_bytes_per_entity(const MeshBase& mesh, const FieldHandle& handle) {
  return mesh.field_bytes_per_entity(handle);
}

const FieldDescriptor* MeshFields::field_descriptor(const MeshBase& mesh, const FieldHandle& handle) {
  return mesh.field_descriptor(handle);
}

// ---- Field queries ----

std::vector<std::string> MeshFields::list_fields(const MeshBase& mesh, EntityKind kind) {
  auto names = mesh.field_names(kind);
  std::sort(names.begin(), names.end());
  return names;
}

FieldHandle MeshFields::get_field_handle(const MeshBase& mesh,
                                        EntityKind kind,
                                        const std::string& name) {
  return mesh.field_handle(kind, name);
}

std::vector<FieldHandle> MeshFields::fields_with_ghost_policy(const MeshBase& mesh,
                                                              FieldGhostPolicy policy) {
  std::vector<FieldHandle> result;

  for (int k = 0; k < 4; ++k) {
    const auto kind = static_cast<EntityKind>(k);
    for (const auto& name : mesh.field_names(kind)) {
      FieldHandle h = mesh.field_handle(kind, name);
      if (h.id == 0) continue;
      const FieldDescriptor* desc = mesh.field_descriptor(h);
      if (!desc) continue;
      if (desc->ghost_policy == policy) {
        result.push_back(h);
      }
    }
  }

  std::sort(result.begin(), result.end(), [](const FieldHandle& a, const FieldHandle& b) {
    const int ak = static_cast<int>(a.kind);
    const int bk = static_cast<int>(b.kind);
    if (ak != bk) return ak < bk;
    return a.name < b.name;
  });

  return result;
}

size_t MeshFields::total_field_count(const MeshBase& mesh) {
  size_t count = 0;
  for (int k = 0; k < 4; ++k) {
    auto fields = list_fields(mesh, static_cast<EntityKind>(k));
    count += fields.size();
  }
  return count;
}

size_t MeshFields::field_memory_usage(const MeshBase& mesh) {
  size_t total = 0;
  for (int k = 0; k < 4; ++k) {
    const auto kind = static_cast<EntityKind>(k);
    const size_t n_entities = entity_count(mesh, kind);
    for (const auto& name : mesh.field_names(kind)) {
      const size_t components = mesh.field_components_by_name(kind, name);
      const size_t bytes_per_comp = mesh.field_bytes_per_component_by_name(kind, name);
      total += n_entities * components * bytes_per_comp;
    }
  }
  return total;
}

// ---- Field operations ----

void MeshFields::copy_field(MeshBase& mesh,
                          const FieldHandle& source,
                          const FieldHandle& target) {
  const void* src_data = field_data(mesh, source);
  void* dst_data = field_data(mesh, target);

  if (!src_data || !dst_data) {
    throw std::runtime_error("Invalid field handles for copy operation");
  }

  const size_t src_components = field_components(mesh, source);
  const size_t dst_components = field_components(mesh, target);

  if (src_components != dst_components) {
    throw std::runtime_error("Component count mismatch in field copy");
  }

  const auto src_type = field_type(mesh, source);
  const auto dst_type = field_type(mesh, target);
  if (src_type != dst_type) {
    throw std::runtime_error("Scalar type mismatch in field copy");
  }

  const size_t src_count = field_entity_count(mesh, source);
  const size_t dst_count = field_entity_count(mesh, target);
  if (src_count != dst_count) {
    throw std::runtime_error("Entity count mismatch in field copy");
  }

  const size_t src_bytes_per = field_bytes_per_entity(mesh, source);
  const size_t dst_bytes_per = field_bytes_per_entity(mesh, target);
  if (src_bytes_per != dst_bytes_per) {
    throw std::runtime_error("Byte stride mismatch in field copy");
  }

  std::memcpy(dst_data, src_data, src_count * src_bytes_per);
  mesh.event_bus().notify(MeshEvent::FieldsChanged);
}

void MeshFields::resize_fields(MeshBase& mesh, EntityKind kind, size_t new_count) {
  mesh.resize_fields(kind, new_count);
}

// ---- Field interpolation ----

void MeshFields::interpolate_cell_to_vertex(MeshBase& mesh,
                                           const FieldHandle& cell_field,
                                           const FieldHandle& vertex_field) {
  if (cell_field.kind != EntityKind::Volume) {
    throw std::invalid_argument("interpolate_cell_to_vertex: cell_field must be a Volume field");
  }
  if (vertex_field.kind != EntityKind::Vertex) {
    throw std::invalid_argument("interpolate_cell_to_vertex: vertex_field must be a Vertex field");
  }
  if (field_type(mesh, cell_field) != FieldScalarType::Float64 ||
      field_type(mesh, vertex_field) != FieldScalarType::Float64) {
    throw std::invalid_argument("interpolate_cell_to_vertex: only Float64 fields are supported");
  }
  if (field_components(mesh, cell_field) != field_components(mesh, vertex_field)) {
    throw std::invalid_argument("interpolate_cell_to_vertex: component count mismatch");
  }

  // Get field data
  const real_t* cell_data = field_data_as<real_t>(mesh, cell_field);
  real_t* vertex_data = field_data_as<real_t>(mesh, vertex_field);

  if (!cell_data || !vertex_data) {
    throw std::runtime_error("Invalid field handles for interpolation");
  }

  size_t n_components = field_components(mesh, cell_field);
  size_t n_vertices = mesh.n_vertices();
  size_t n_cells = mesh.n_cells();

  // Initialize vertex data to zero
  std::fill(vertex_data, vertex_data + n_vertices * n_components, 0.0);

  // Count contributions per vertex
  std::vector<size_t> vertex_counts(n_vertices, 0);

  // Accumulate cell values at vertices
  for (size_t c = 0; c < n_cells; ++c) {
    auto [vertices_ptr, n_cell_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));

    for (size_t i = 0; i < n_cell_vertices; ++i) {
      index_t vertex_id = vertices_ptr[i];
      if (vertex_id < 0 || static_cast<size_t>(vertex_id) >= n_vertices) {
        throw std::runtime_error("interpolate_cell_to_vertex: cell connectivity contains invalid vertex index");
      }

      // Add cell value to vertex
      for (size_t comp = 0; comp < n_components; ++comp) {
        vertex_data[vertex_id * n_components + comp] +=
          cell_data[c * n_components + comp];
      }

      vertex_counts[vertex_id]++;
    }
  }

  // Average the accumulated values
  for (size_t v = 0; v < n_vertices; ++v) {
    if (vertex_counts[v] > 0) {
      for (size_t comp = 0; comp < n_components; ++comp) {
        vertex_data[v * n_components + comp] /= static_cast<real_t>(vertex_counts[v]);
      }
    }
  }

  mesh.event_bus().notify(MeshEvent::FieldsChanged);
}

void MeshFields::interpolate_vertex_to_cell(MeshBase& mesh,
                                           const FieldHandle& vertex_field,
                                           const FieldHandle& cell_field) {
  if (vertex_field.kind != EntityKind::Vertex) {
    throw std::invalid_argument("interpolate_vertex_to_cell: vertex_field must be a Vertex field");
  }
  if (cell_field.kind != EntityKind::Volume) {
    throw std::invalid_argument("interpolate_vertex_to_cell: cell_field must be a Volume field");
  }
  if (field_type(mesh, vertex_field) != FieldScalarType::Float64 ||
      field_type(mesh, cell_field) != FieldScalarType::Float64) {
    throw std::invalid_argument("interpolate_vertex_to_cell: only Float64 fields are supported");
  }
  if (field_components(mesh, vertex_field) != field_components(mesh, cell_field)) {
    throw std::invalid_argument("interpolate_vertex_to_cell: component count mismatch");
  }

  // Get field data
  const real_t* vertex_data = field_data_as<real_t>(mesh, vertex_field);
  real_t* cell_data = field_data_as<real_t>(mesh, cell_field);

  if (!vertex_data || !cell_data) {
    throw std::runtime_error("Invalid field handles for interpolation");
  }

  size_t n_components = field_components(mesh, vertex_field);
  size_t n_cells = mesh.n_cells();

  // Average vertex values for each cell
  for (size_t c = 0; c < n_cells; ++c) {
    auto [vertices_ptr, n_cell_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));

    // Initialize cell value
    for (size_t comp = 0; comp < n_components; ++comp) {
      cell_data[c * n_components + comp] = 0.0;
    }

    // Sum vertex values
    for (size_t i = 0; i < n_cell_vertices; ++i) {
      index_t vertex_id = vertices_ptr[i];
      if (vertex_id < 0 || static_cast<size_t>(vertex_id) >= mesh.n_vertices()) {
        throw std::runtime_error("interpolate_vertex_to_cell: cell connectivity contains invalid vertex index");
      }
      for (size_t comp = 0; comp < n_components; ++comp) {
        cell_data[c * n_components + comp] +=
          vertex_data[vertex_id * n_components + comp];
      }
    }

    // Average
    if (n_cell_vertices > 0) {
      for (size_t comp = 0; comp < n_components; ++comp) {
        cell_data[c * n_components + comp] /= static_cast<real_t>(n_cell_vertices);
      }
    }
  }

  mesh.event_bus().notify(MeshEvent::FieldsChanged);
}

void MeshFields::restrict_field(const MeshBase& fine_mesh,
                                MeshBase& coarse_mesh,
                                const FieldHandle& fine_field,
                                const FieldHandle& coarse_field) {
  const size_t fine_count = fine_mesh.field_entity_count(fine_field);
  const size_t coarse_count = coarse_mesh.field_entity_count(coarse_field);
  if (fine_count != coarse_count) {
    throw std::runtime_error("restrict_field: entity mapping required (counts differ); use overload taking EntityTransferMap");
  }
  EntityTransferMap map = EntityTransferMap::identity(fine_field.kind, fine_count);
  restrict_field(fine_mesh, coarse_mesh, fine_field, coarse_field, map);
}

void MeshFields::prolongate_field(const MeshBase& coarse_mesh,
                                  MeshBase& fine_mesh,
                                  const FieldHandle& coarse_field,
                                  const FieldHandle& fine_field) {
  const size_t coarse_count = coarse_mesh.field_entity_count(coarse_field);
  const size_t fine_count = fine_mesh.field_entity_count(fine_field);
  if (coarse_count != fine_count) {
    throw std::runtime_error("prolongate_field: entity mapping required (counts differ); use overload taking EntityTransferMap");
  }
  EntityTransferMap map = EntityTransferMap::identity(coarse_field.kind, coarse_count);
  prolongate_field(coarse_mesh, fine_mesh, coarse_field, fine_field, map);
}

void MeshFields::restrict_field(const MeshBase& fine_mesh,
                                MeshBase& coarse_mesh,
                                const FieldHandle& fine_field,
                                const FieldHandle& coarse_field,
                                const EntityTransferMap& map) {
  transfer_field_with_map(fine_mesh, coarse_mesh, fine_field, coarse_field, map);
}

void MeshFields::prolongate_field(const MeshBase& coarse_mesh,
                                  MeshBase& fine_mesh,
                                  const FieldHandle& coarse_field,
                                  const FieldHandle& fine_field,
                                  const EntityTransferMap& map) {
  transfer_field_with_map(coarse_mesh, fine_mesh, coarse_field, fine_field, map);
}

EntityTransferMap MeshFields::make_volume_weighted_cell_restriction_map(
    const MeshBase& fine_mesh,
    const MeshBase& coarse_mesh,
    const std::vector<std::vector<index_t>>& coarse_to_fine_cells,
    Configuration cfg) {
  if (coarse_to_fine_cells.size() != coarse_mesh.n_cells()) {
    throw std::invalid_argument("make_volume_weighted_cell_restriction_map: coarse_to_fine_cells size mismatch");
  }

  std::vector<std::vector<real_t>> weights(coarse_to_fine_cells.size());
  for (size_t c = 0; c < coarse_to_fine_cells.size(); ++c) {
    const auto& children = coarse_to_fine_cells[c];
    if (children.empty()) {
      throw std::invalid_argument("make_volume_weighted_cell_restriction_map: empty child list for coarse cell");
    }
    auto& w = weights[c];
    w.reserve(children.size());
    for (index_t child : children) {
      if (child < 0 || static_cast<size_t>(child) >= fine_mesh.n_cells()) {
        throw std::invalid_argument("make_volume_weighted_cell_restriction_map: fine cell index out of range");
      }
      w.push_back(fine_mesh.cell_measure(child, cfg));
    }
  }

  EntityTransferMap map = EntityTransferMap::from_lists(
      EntityKind::Volume, fine_mesh.n_cells(), coarse_to_fine_cells, &weights);
  map.normalize_weights(0.0);
  map.dst_count = coarse_mesh.n_cells();
  map.validate(true);
  return map;
}

// ---- Field statistics ----

MeshFields::FieldStats MeshFields::compute_stats(const MeshBase& mesh,
                                                const FieldHandle& handle,
                                                size_t component) {
  const void* raw = field_data(mesh, handle);
  if (!raw) {
    throw std::invalid_argument("compute_stats: invalid field handle");
  }

  const size_t n_entities = field_entity_count(mesh, handle);
  const size_t n_components = field_components(mesh, handle);

  if (component >= n_components) {
    throw std::out_of_range("Component index out of range");
  }

  const auto type = field_type(mesh, handle);
  switch (type) {
    case FieldScalarType::Int32:
      return compute_stats_impl(reinterpret_cast<const int32_t*>(raw), n_entities, n_components, component);
    case FieldScalarType::Int64:
      return compute_stats_impl(reinterpret_cast<const int64_t*>(raw), n_entities, n_components, component);
    case FieldScalarType::Float32:
      return compute_stats_impl(reinterpret_cast<const float*>(raw), n_entities, n_components, component);
    case FieldScalarType::Float64:
      return compute_stats_impl(reinterpret_cast<const double*>(raw), n_entities, n_components, component);
    case FieldScalarType::UInt8:
      return compute_stats_impl(reinterpret_cast<const uint8_t*>(raw), n_entities, n_components, component);
    case FieldScalarType::Custom:
      break;
  }

  throw std::invalid_argument("compute_stats: unsupported scalar type");
}

real_t MeshFields::compute_l2_norm(const MeshBase& mesh,
                                  const FieldHandle& handle) {
  const void* raw = field_data(mesh, handle);
  if (!raw) {
    throw std::invalid_argument("compute_l2_norm: invalid field handle");
  }

  const size_t n_entities = field_entity_count(mesh, handle);
  const size_t n_components = field_components(mesh, handle);
  const size_t n_total = n_entities * n_components;

  const auto type = field_type(mesh, handle);
  switch (type) {
    case FieldScalarType::Int32:
      return compute_l2_norm_impl(reinterpret_cast<const int32_t*>(raw), n_total);
    case FieldScalarType::Int64:
      return compute_l2_norm_impl(reinterpret_cast<const int64_t*>(raw), n_total);
    case FieldScalarType::Float32:
      return compute_l2_norm_impl(reinterpret_cast<const float*>(raw), n_total);
    case FieldScalarType::Float64:
      return compute_l2_norm_impl(reinterpret_cast<const double*>(raw), n_total);
    case FieldScalarType::UInt8:
      return compute_l2_norm_impl(reinterpret_cast<const uint8_t*>(raw), n_total);
    case FieldScalarType::Custom:
      break;
  }

  throw std::invalid_argument("compute_l2_norm: unsupported scalar type");
}

real_t MeshFields::compute_inf_norm(const MeshBase& mesh,
                                   const FieldHandle& handle) {
  const void* raw = field_data(mesh, handle);
  if (!raw) {
    throw std::invalid_argument("compute_inf_norm: invalid field handle");
  }

  const size_t n_entities = field_entity_count(mesh, handle);
  const size_t n_components = field_components(mesh, handle);
  const size_t n_total = n_entities * n_components;

  const auto type = field_type(mesh, handle);
  switch (type) {
    case FieldScalarType::Int32:
      return compute_inf_norm_impl(reinterpret_cast<const int32_t*>(raw), n_total);
    case FieldScalarType::Int64:
      return compute_inf_norm_impl(reinterpret_cast<const int64_t*>(raw), n_total);
    case FieldScalarType::Float32:
      return compute_inf_norm_impl(reinterpret_cast<const float*>(raw), n_total);
    case FieldScalarType::Float64:
      return compute_inf_norm_impl(reinterpret_cast<const double*>(raw), n_total);
    case FieldScalarType::UInt8:
      return compute_inf_norm_impl(reinterpret_cast<const uint8_t*>(raw), n_total);
    case FieldScalarType::Custom:
      break;
  }

  throw std::invalid_argument("compute_inf_norm: unsupported scalar type");
}

} // namespace svmp
