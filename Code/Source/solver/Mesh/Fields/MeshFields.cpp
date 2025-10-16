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

// ---- Field queries ----

std::vector<std::string> MeshFields::list_fields(const MeshBase& mesh, EntityKind kind) {
  // This would need access to MeshBase internals
  // For now, return empty vector - would need to expose this in MeshBase
  return {};
}

FieldHandle MeshFields::get_field_handle(const MeshBase& mesh,
                                        EntityKind kind,
                                        const std::string& name) {
  // Create a handle and check if field exists
  FieldHandle handle;
  handle.kind = kind;
  handle.name = name;

  if (mesh.has_field(kind, name)) {
    // Would need to get actual handle ID from MeshBase
    handle.id = 0;  // Placeholder
  }

  return handle;
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
  // Would need to iterate through all fields and sum their sizes
  // This requires access to MeshBase internals
  return 0;  // Placeholder
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

  size_t src_components = field_components(mesh, source);
  size_t dst_components = field_components(mesh, target);

  if (src_components != dst_components) {
    throw std::runtime_error("Component count mismatch in field copy");
  }

  size_t count = field_entity_count(mesh, source);
  size_t bytes_per = field_bytes_per_entity(mesh, source);

  std::memcpy(dst_data, src_data, count * bytes_per);
}

void MeshFields::resize_fields(MeshBase& mesh, EntityKind kind, size_t new_count) {
  // This would need to be implemented in MeshBase to resize field storage
  // For all fields attached to the given entity kind
}

// ---- Field interpolation ----

void MeshFields::interpolate_cell_to_vertex(const MeshBase& mesh,
                                           const FieldHandle& cell_field,
                                           const FieldHandle& node_field) {
  // Get field data
  const real_t* cell_data = field_data_as<real_t>(mesh, cell_field);
  real_t* node_data = const_cast<real_t*>(field_data_as<real_t>(mesh, node_field));

  if (!cell_data || !node_data) {
    throw std::runtime_error("Invalid field handles for interpolation");
  }

  size_t n_components = field_components(mesh, cell_field);
  size_t n_vertices = mesh.n_vertices();
  size_t n_cells = mesh.n_cells();

  // Initialize vertex data to zero
  std::fill(node_data, node_data + n_vertices * n_components, 0.0);

  // Count contributions per vertex
  std::vector<size_t> vertex_counts(n_vertices, 0);

  // Accumulate cell values at vertices
  for (size_t c = 0; c < n_cells; ++c) {
    auto [vertices_ptr, n_cell_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));

    for (size_t i = 0; i < n_cell_vertices; ++i) {
      index_t vertex_id = vertices_ptr[i];

      // Add cell value to vertex
      for (size_t comp = 0; comp < n_components; ++comp) {
        node_data[vertex_id * n_components + comp] +=
          cell_data[c * n_components + comp];
      }

      vertex_counts[vertex_id]++;
    }
  }

  // Average the accumulated values
  for (size_t v = 0; v < n_vertices; ++v) {
    if (vertex_counts[v] > 0) {
      for (size_t comp = 0; comp < n_components; ++comp) {
        node_data[v * n_components + comp] /= static_cast<real_t>(vertex_counts[v]);
      }
    }
  }
}

void MeshFields::interpolate_vertex_to_cell(const MeshBase& mesh,
                                           const FieldHandle& node_field,
                                           const FieldHandle& cell_field) {
  // Get field data
  const real_t* node_data = field_data_as<real_t>(mesh, node_field);
  real_t* cell_data = const_cast<real_t*>(field_data_as<real_t>(mesh, cell_field));

  if (!node_data || !cell_data) {
    throw std::runtime_error("Invalid field handles for interpolation");
  }

  size_t n_components = field_components(mesh, node_field);
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
      for (size_t comp = 0; comp < n_components; ++comp) {
        cell_data[c * n_components + comp] +=
          node_data[vertex_id * n_components + comp];
      }
    }

    // Average
    if (n_cell_vertices > 0) {
      for (size_t comp = 0; comp < n_components; ++comp) {
        cell_data[c * n_components + comp] /= static_cast<real_t>(n_cell_vertices);
      }
    }
  }
}

void MeshFields::restrict_field(const MeshBase& fine_mesh,
                              const MeshBase& coarse_mesh,
                              const FieldHandle& fine_field,
                              const FieldHandle& coarse_field) {
  // Restriction from fine to coarse mesh
  // This would require a mapping between fine and coarse entities
  // Implementation depends on the refinement hierarchy
  throw std::runtime_error("Field restriction not yet implemented");
}

void MeshFields::prolongate_field(const MeshBase& coarse_mesh,
                                 const MeshBase& fine_mesh,
                                 const FieldHandle& coarse_field,
                                 const FieldHandle& fine_field) {
  // Prolongation from coarse to fine mesh
  // This would require a mapping between coarse and fine entities
  // Implementation depends on the refinement hierarchy
  throw std::runtime_error("Field prolongation not yet implemented");
}

// ---- Field statistics ----

MeshFields::FieldStats MeshFields::compute_stats(const MeshBase& mesh,
                                                const FieldHandle& handle,
                                                size_t component) {
  FieldStats stats;

  const real_t* data = field_data_as<real_t>(mesh, handle);
  if (!data) return stats;

  size_t n_entities = field_entity_count(mesh, handle);
  size_t n_components = field_components(mesh, handle);

  if (component >= n_components) {
    throw std::out_of_range("Component index out of range");
  }

  if (n_entities == 0) return stats;

  // Initialize with first value
  stats.min = data[component];
  stats.max = data[component];
  stats.sum = 0;

  // First pass: compute min, max, sum
  for (size_t i = 0; i < n_entities; ++i) {
    real_t val = data[i * n_components + component];
    stats.min = std::min(stats.min, val);
    stats.max = std::max(stats.max, val);
    stats.sum += val;
  }

  stats.mean = stats.sum / static_cast<real_t>(n_entities);

  // Second pass: compute standard deviation
  real_t sum_sq_diff = 0;
  for (size_t i = 0; i < n_entities; ++i) {
    real_t val = data[i * n_components + component];
    real_t diff = val - stats.mean;
    sum_sq_diff += diff * diff;
  }

  stats.std_dev = std::sqrt(sum_sq_diff / static_cast<real_t>(n_entities));

  return stats;
}

real_t MeshFields::compute_l2_norm(const MeshBase& mesh,
                                  const FieldHandle& handle) {
  const real_t* data = field_data_as<real_t>(mesh, handle);
  if (!data) return 0.0;

  size_t n_entities = field_entity_count(mesh, handle);
  size_t n_components = field_components(mesh, handle);

  real_t sum_sq = 0;
  for (size_t i = 0; i < n_entities * n_components; ++i) {
    sum_sq += data[i] * data[i];
  }

  return std::sqrt(sum_sq);
}

real_t MeshFields::compute_inf_norm(const MeshBase& mesh,
                                   const FieldHandle& handle) {
  const real_t* data = field_data_as<real_t>(mesh, handle);
  if (!data) return 0.0;

  size_t n_entities = field_entity_count(mesh, handle);
  size_t n_components = field_components(mesh, handle);

  real_t max_val = 0;
  for (size_t i = 0; i < n_entities * n_components; ++i) {
    max_val = std::max(max_val, std::abs(data[i]));
  }

  return max_val;
}

} // namespace svmp
