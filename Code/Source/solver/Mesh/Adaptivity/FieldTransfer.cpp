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

#include "FieldTransfer.h"
#include "../Core/MeshBase.h"
#include "../Fields/MeshFields.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace svmp {

namespace {

bool should_transfer_field_name(const AdaptivityOptions& options, const std::string& name) {
  if (!options.transfer_fields.empty()) {
    bool selected = false;
    for (const auto& n : options.transfer_fields) {
      if (n == name) {
        selected = true;
        break;
      }
    }
    if (!selected) return false;
  }

  for (const auto& n : options.skip_fields) {
    if (n == name) return false;
  }
  return true;
}

template <typename T>
bool is_near_constant(const std::vector<T>& v, T tol) {
  if (v.empty()) return true;
  const T first = v.front();
  for (const T& x : v) {
    if (std::abs(x - first) > tol) return false;
  }
  return true;
}

template <typename AccT>
double weighted_sum_double(const double* src,
                           const std::vector<std::pair<size_t, double>>& weights,
                           size_t src_count) {
  AccT sum = 0;
  for (const auto& kv : weights) {
    const size_t si = kv.first;
    const double w = kv.second;
    if (si >= src_count) continue;
    sum += static_cast<AccT>(w) * static_cast<AccT>(src[si]);
  }
  return static_cast<double>(sum);
}

template <typename ScalarT>
void transfer_vertex_numeric_by_weights_or_gid(const MeshBase& old_mesh,
                                               MeshBase& new_mesh,
                                               const FieldHandle& old_h,
                                               const FieldHandle& new_h,
                                               const ParentChildMap& map,
                                               bool round_integral) {
  const size_t old_n = old_mesh.n_vertices();
  const size_t new_n = new_mesh.n_vertices();
  const size_t components = MeshFields::field_components(old_mesh, old_h);

  const auto* old_data = MeshFields::field_data_as<const ScalarT>(old_mesh, old_h);
  auto* new_data = MeshFields::field_data_as<ScalarT>(new_mesh, new_h);
  if (!old_data || !new_data) {
    throw std::runtime_error("FieldTransfer: invalid field handles");
  }

  // Prefer explicit interpolation weights when provided.
  if (!map.child_vertex_weights.empty()) {
    for (size_t v = 0; v < new_n; ++v) {
      const auto it = map.child_vertex_weights.find(v);
      if (it == map.child_vertex_weights.end()) {
        // Fallback: copy by GID if possible.
        const gid_t g = new_mesh.vertex_gids().at(v);
        const index_t ov = old_mesh.global_to_local_vertex(g);
        for (size_t c = 0; c < components; ++c) {
          if (ov != INVALID_INDEX) {
            new_data[v * components + c] =
                old_data[static_cast<size_t>(ov) * components + c];
          } else {
            new_data[v * components + c] = ScalarT{};
          }
        }
        continue;
      }

      for (size_t c = 0; c < components; ++c) {
        long double acc = 0;
        for (const auto& kv : it->second) {
          const size_t si = kv.first;
          if (si >= old_n) continue;
          acc += static_cast<long double>(kv.second) *
                 static_cast<long double>(old_data[si * components + c]);
        }
        if (round_integral && !std::is_floating_point_v<ScalarT>) {
          new_data[v * components + c] = static_cast<ScalarT>(std::llround(acc));
        } else {
          new_data[v * components + c] = static_cast<ScalarT>(acc);
        }
      }
    }
    return;
  }

  // Restriction map: parent (coarse) vertex -> contributing fine vertices.
  if (!map.parent_vertex_to_children.empty()) {
    for (size_t v = 0; v < new_n; ++v) {
      const auto it = map.parent_vertex_to_children.find(v);
      if (it == map.parent_vertex_to_children.end() || it->second.empty()) {
        // Fallback: copy by GID if possible.
        const gid_t g = new_mesh.vertex_gids().at(v);
        const index_t ov = old_mesh.global_to_local_vertex(g);
        for (size_t c = 0; c < components; ++c) {
          if (ov != INVALID_INDEX) {
            new_data[v * components + c] =
                old_data[static_cast<size_t>(ov) * components + c];
          } else {
            new_data[v * components + c] = ScalarT{};
          }
        }
        continue;
      }

      for (size_t c = 0; c < components; ++c) {
        long double acc = 0;
        size_t count = 0;
        for (size_t si : it->second) {
          if (si >= old_n) continue;
          acc += static_cast<long double>(old_data[si * components + c]);
          ++count;
        }
        if (count == 0) {
          new_data[v * components + c] = ScalarT{};
        } else {
          const long double avg = acc / static_cast<long double>(count);
          if (round_integral && !std::is_floating_point_v<ScalarT>) {
            new_data[v * components + c] = static_cast<ScalarT>(std::llround(avg));
          } else {
            new_data[v * components + c] = static_cast<ScalarT>(avg);
          }
        }
      }
    }
    return;
  }

  // Default: copy by GID (zeros for new vertices).
  for (size_t v = 0; v < new_n; ++v) {
    const gid_t g = new_mesh.vertex_gids().at(v);
    const index_t ov = old_mesh.global_to_local_vertex(g);
    for (size_t c = 0; c < components; ++c) {
      if (ov != INVALID_INDEX) {
        new_data[v * components + c] =
            old_data[static_cast<size_t>(ov) * components + c];
      } else {
        new_data[v * components + c] = ScalarT{};
      }
    }
  }
}

void transfer_vertex_custom_by_gid(const MeshBase& old_mesh,
                                   MeshBase& new_mesh,
                                   const FieldHandle& old_h,
                                   const FieldHandle& new_h) {
  const size_t old_n = old_mesh.n_vertices();
  const size_t new_n = new_mesh.n_vertices();

  const auto* old_data = MeshFields::field_data_as<const uint8_t>(old_mesh, old_h);
  auto* new_data = MeshFields::field_data_as<uint8_t>(new_mesh, new_h);
  if (!old_data || !new_data) {
    throw std::runtime_error("FieldTransfer: invalid custom field handles");
  }

  const size_t bytes_per = MeshFields::field_bytes_per_entity(new_mesh, new_h);
  std::fill(new_data, new_data + new_n * bytes_per, 0);

  for (size_t v = 0; v < new_n; ++v) {
    const gid_t g = new_mesh.vertex_gids().at(v);
    const index_t ov = old_mesh.global_to_local_vertex(g);
    if (ov == INVALID_INDEX) continue;
    const uint8_t* src = old_data + static_cast<size_t>(ov) * bytes_per;
    uint8_t* dst = new_data + v * bytes_per;
    std::memcpy(dst, src, bytes_per);
  }

  (void)old_n;
}

template <typename ScalarT>
void transfer_cell_numeric(const MeshBase& old_mesh,
                           MeshBase& new_mesh,
                           const FieldHandle& old_h,
                           const FieldHandle& new_h,
                           const ParentChildMap& map,
                           bool average_for_coarsening) {
  const size_t old_n = old_mesh.n_cells();
  const size_t new_n = new_mesh.n_cells();
  const size_t components = MeshFields::field_components(old_mesh, old_h);

  const auto* old_data = MeshFields::field_data_as<const ScalarT>(old_mesh, old_h);
  auto* new_data = MeshFields::field_data_as<ScalarT>(new_mesh, new_h);
  if (!old_data || !new_data) {
    throw std::runtime_error("FieldTransfer: invalid cell field handles");
  }

  // Refinement: child cell -> parent cell index.
  if (!map.child_to_parent.empty() && map.child_to_parent.size() == new_n) {
    for (size_t c = 0; c < new_n; ++c) {
      const size_t p = map.child_to_parent[c];
      for (size_t k = 0; k < components; ++k) {
        if (p < old_n) {
          new_data[c * components + k] = old_data[p * components + k];
        } else {
          new_data[c * components + k] = ScalarT{};
        }
      }
    }
    return;
  }

  // Coarsening: parent cell -> children indices.
  if (!map.parent_to_children.empty()) {
    for (size_t c = 0; c < new_n; ++c) {
      const auto it = map.parent_to_children.find(c);
      if (it == map.parent_to_children.end() || it->second.empty()) {
        // Fallback to GID copy.
        const gid_t g = new_mesh.cell_gids().at(c);
        const index_t oc = old_mesh.global_to_local_cell(g);
        for (size_t k = 0; k < components; ++k) {
          if (oc != INVALID_INDEX) {
            new_data[c * components + k] = old_data[static_cast<size_t>(oc) * components + k];
          } else {
            new_data[c * components + k] = ScalarT{};
          }
        }
        continue;
      }

      if (average_for_coarsening && std::is_floating_point_v<ScalarT>) {
        for (size_t k = 0; k < components; ++k) {
          long double acc = 0;
          size_t count = 0;
          for (size_t child : it->second) {
            if (child >= old_n) continue;
            acc += static_cast<long double>(old_data[child * components + k]);
            ++count;
          }
          new_data[c * components + k] = (count == 0) ? ScalarT{} : static_cast<ScalarT>(acc / static_cast<long double>(count));
        }
      } else {
        const size_t rep = *std::min_element(it->second.begin(), it->second.end());
        for (size_t k = 0; k < components; ++k) {
          if (rep < old_n) {
            new_data[c * components + k] = old_data[rep * components + k];
          } else {
            new_data[c * components + k] = ScalarT{};
          }
        }
      }
    }
    return;
  }

  // Default: copy by GID.
  for (size_t c = 0; c < new_n; ++c) {
    const gid_t g = new_mesh.cell_gids().at(c);
    const index_t oc = old_mesh.global_to_local_cell(g);
    for (size_t k = 0; k < components; ++k) {
      if (oc != INVALID_INDEX) {
        new_data[c * components + k] = old_data[static_cast<size_t>(oc) * components + k];
      } else {
        new_data[c * components + k] = ScalarT{};
      }
    }
  }
}

void transfer_cell_custom(const MeshBase& old_mesh,
                          MeshBase& new_mesh,
                          const FieldHandle& old_h,
                          const FieldHandle& new_h,
                          const ParentChildMap& map) {
  const size_t old_n = old_mesh.n_cells();
  const size_t new_n = new_mesh.n_cells();

  const auto* old_data = MeshFields::field_data_as<const uint8_t>(old_mesh, old_h);
  auto* new_data = MeshFields::field_data_as<uint8_t>(new_mesh, new_h);
  if (!old_data || !new_data) {
    throw std::runtime_error("FieldTransfer: invalid custom cell field handles");
  }

  const size_t bytes_per = MeshFields::field_bytes_per_entity(new_mesh, new_h);
  std::fill(new_data, new_data + new_n * bytes_per, 0);

  // Refinement: child cell -> parent cell index.
  if (!map.child_to_parent.empty() && map.child_to_parent.size() == new_n) {
    for (size_t c = 0; c < new_n; ++c) {
      const size_t p = map.child_to_parent[c];
      if (p >= old_n) continue;
      std::memcpy(new_data + c * bytes_per, old_data + p * bytes_per, bytes_per);
    }
    return;
  }

  // Coarsening: parent cell -> representative child.
  if (!map.parent_to_children.empty()) {
    for (size_t c = 0; c < new_n; ++c) {
      const auto it = map.parent_to_children.find(c);
      if (it == map.parent_to_children.end() || it->second.empty()) continue;
      const size_t rep = *std::min_element(it->second.begin(), it->second.end());
      if (rep >= old_n) continue;
      std::memcpy(new_data + c * bytes_per, old_data + rep * bytes_per, bytes_per);
    }
    return;
  }

  // Default: copy by GID.
  for (size_t c = 0; c < new_n; ++c) {
    const gid_t g = new_mesh.cell_gids().at(c);
    const index_t oc = old_mesh.global_to_local_cell(g);
    if (oc == INVALID_INDEX) continue;
    std::memcpy(new_data + c * bytes_per, old_data + static_cast<size_t>(oc) * bytes_per, bytes_per);
  }
}

} // namespace

LinearInterpolationTransfer::LinearInterpolationTransfer(const Config& config)
    : config_(config) {}

TransferStats LinearInterpolationTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  (void)old_fields;
  (void)new_fields;

  auto start = std::chrono::high_resolution_clock::now();

  MeshBase& writable_new_mesh = const_cast<MeshBase&>(new_mesh);
  TransferStats stats;

  const std::vector<EntityKind> kinds = {EntityKind::Vertex, EntityKind::Volume};
  for (EntityKind kind : kinds) {
    for (const auto& name : old_mesh.field_names(kind)) {
      if (!should_transfer_field_name(options, name)) continue;

      const FieldHandle old_h = MeshFields::get_field_handle(old_mesh, kind, name);
      if (old_h.id == 0) continue;

      const auto type = MeshFields::field_type(old_mesh, old_h);
      const size_t components = MeshFields::field_components(old_mesh, old_h);
      const size_t custom_bpc =
          (type == FieldScalarType::Custom)
              ? old_mesh.field_bytes_per_component_by_name(kind, name)
              : 0u;

      FieldHandle new_h = MeshFields::attach_field(writable_new_mesh, kind, name, type, components, custom_bpc);

      if (const FieldDescriptor* desc = old_mesh.field_descriptor(old_h)) {
        writable_new_mesh.set_field_descriptor(new_h, *desc);
      }

      if (kind == EntityKind::Vertex) {
        switch (type) {
          case FieldScalarType::Float64:
            transfer_vertex_numeric_by_weights_or_gid<double>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/false);
            stats.num_prolongations++;
            break;
          case FieldScalarType::Float32:
            transfer_vertex_numeric_by_weights_or_gid<float>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/false);
            stats.num_prolongations++;
            break;
          case FieldScalarType::Int32:
            transfer_vertex_numeric_by_weights_or_gid<int32_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/true);
            stats.num_prolongations++;
            break;
          case FieldScalarType::Int64:
            transfer_vertex_numeric_by_weights_or_gid<int64_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/true);
            stats.num_prolongations++;
            break;
          case FieldScalarType::UInt8:
            transfer_vertex_numeric_by_weights_or_gid<uint8_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/true);
            stats.num_prolongations++;
            break;
          case FieldScalarType::Custom:
            transfer_vertex_custom_by_gid(old_mesh, writable_new_mesh, old_h, new_h);
            stats.num_prolongations++;
            break;
        }
      } else if (kind == EntityKind::Volume) {
        switch (type) {
          case FieldScalarType::Float64:
            transfer_cell_numeric<double>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/true);
            stats.num_restrictions++;
            break;
          case FieldScalarType::Float32:
            transfer_cell_numeric<float>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/true);
            stats.num_restrictions++;
            break;
          case FieldScalarType::Int32:
            transfer_cell_numeric<int32_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            stats.num_restrictions++;
            break;
          case FieldScalarType::Int64:
            transfer_cell_numeric<int64_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            stats.num_restrictions++;
            break;
          case FieldScalarType::UInt8:
            transfer_cell_numeric<uint8_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            stats.num_restrictions++;
            break;
          case FieldScalarType::Custom:
            transfer_cell_custom(old_mesh, writable_new_mesh, old_h, new_h, parent_child);
            stats.num_restrictions++;
            break;
        }
      }

      stats.num_fields++;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  stats.transfer_time = std::chrono::duration<double>(end - start).count();
  return stats;
}

void LinearInterpolationTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)old_mesh;
  (void)new_mesh;

  if (new_field.empty()) {
    new_field.resize(old_field.size(), 0.0);
  }

  if (!parent_child.child_vertex_weights.empty()) {
    for (size_t v = 0; v < new_field.size(); ++v) {
      const auto it = parent_child.child_vertex_weights.find(v);
      if (it == parent_child.child_vertex_weights.end()) {
        if (v < old_field.size()) {
          new_field[v] = old_field[v];
        }
        continue;
      }
      new_field[v] = weighted_sum_double<long double>(old_field.data(), it->second, old_field.size());
    }
    return;
  }

  // No weights: preserve constant fields, otherwise do a best-effort prefix copy.
  if (!old_field.empty() && is_near_constant(old_field, 1e-14)) {
    std::fill(new_field.begin(), new_field.end(), old_field.front());
    return;
  }

  const size_t n = std::min(old_field.size(), new_field.size());
  std::copy(old_field.begin(), old_field.begin() + static_cast<ptrdiff_t>(n), new_field.begin());
  if (new_field.size() > n) {
    const double fill = old_field.empty() ? 0.0 : old_field.back();
    std::fill(new_field.begin() + static_cast<ptrdiff_t>(n), new_field.end(), fill);
  }
}

void LinearInterpolationTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)old_mesh;
  (void)new_mesh;

  if (new_field.empty()) {
    new_field.resize(old_field.size(), 0.0);
  }

  if (!parent_child.parent_vertex_to_children.empty()) {
    for (size_t v = 0; v < new_field.size(); ++v) {
      const auto it = parent_child.parent_vertex_to_children.find(v);
      if (it == parent_child.parent_vertex_to_children.end() || it->second.empty()) {
        if (v < old_field.size()) new_field[v] = old_field[v];
        continue;
      }
      // If the map provides an explicit injection (child -> parent) mapping for a
      // particular fine vertex, prefer it to preserve linear fields exactly at
      // coincident vertices (e.g., 1D uniform refine/coarsen round-trip tests).
      bool injected = false;
      if (!parent_child.child_vertex_weights.empty()) {
        for (size_t si : it->second) {
          const auto wit = parent_child.child_vertex_weights.find(si);
          if (wit == parent_child.child_vertex_weights.end()) continue;
          if (wit->second.size() != 1) continue;
          const auto [pi, w] = wit->second.front();
          if (pi == v && std::abs(w - 1.0) < 1e-14) {
            if (si < old_field.size()) {
              new_field[v] = old_field[si];
              injected = true;
              break;
            }
          }
        }
      }
      if (injected) continue;
      long double acc = 0;
      size_t count = 0;
      for (size_t si : it->second) {
        if (si >= old_field.size()) continue;
        acc += static_cast<long double>(old_field[si]);
        ++count;
      }
      new_field[v] = (count == 0) ? 0.0 : static_cast<double>(acc / static_cast<long double>(count));
    }
    return;
  }

  // Fallback: preserve constants, otherwise prefix copy.
  if (!old_field.empty() && is_near_constant(old_field, 1e-14)) {
    std::fill(new_field.begin(), new_field.end(), old_field.front());
    return;
  }

  const size_t n = std::min(old_field.size(), new_field.size());
  std::copy(old_field.begin(), old_field.begin() + static_cast<ptrdiff_t>(n), new_field.begin());
}

double LinearInterpolationTransfer::interpolate_at_vertex(
    const std::vector<double>& old_field,
    const std::vector<std::pair<size_t, double>>& weights) const {
  double value = 0.0;
  for (const auto& [idx, w] : weights) {
    if (idx < old_field.size()) {
      value += w * old_field[idx];
    }
  }
  return value;
}

double LinearInterpolationTransfer::average_from_children(
    const std::vector<double>& old_field,
    const std::vector<size_t>& children,
    const MeshBase& mesh) const {
  (void)mesh;
  if (children.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  size_t count = 0;
  for (auto c : children) {
    if (c < old_field.size()) {
      sum += old_field[c];
      ++count;
    }
  }
  return count > 0 ? (sum / static_cast<double>(count)) : 0.0;
}

ConservativeTransfer::ConservativeTransfer(const Config& config)
    : config_(config) {}

TransferStats ConservativeTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  LinearInterpolationTransfer::Config cfg;
  LinearInterpolationTransfer linear(cfg);
  return linear.transfer(old_mesh, new_mesh, old_fields, new_fields, parent_child, options);
}

void ConservativeTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  LinearInterpolationTransfer::Config cfg;
  LinearInterpolationTransfer linear(cfg);
  linear.prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
  enforce_conservation(old_mesh, new_mesh, old_field, new_field);
}

void ConservativeTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

double ConservativeTransfer::compute_integral(
    const MeshBase& mesh,
    const std::vector<double>& field) const {
  (void)mesh;
  return std::accumulate(field.begin(), field.end(), 0.0);
}

void ConservativeTransfer::enforce_conservation(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field) const {
  (void)old_mesh;
  (void)new_mesh;
  if (new_field.empty()) return;

  const double old_sum = std::accumulate(old_field.begin(), old_field.end(), 0.0);
  const double new_sum = std::accumulate(new_field.begin(), new_field.end(), 0.0);

  if (std::abs(new_sum) > config_.conservation_tolerance) {
    const double scale = old_sum / new_sum;
    for (double& x : new_field) x *= scale;
  } else {
    const double fill = old_sum / static_cast<double>(new_field.size());
    std::fill(new_field.begin(), new_field.end(), fill);
  }
}

std::vector<double> ConservativeTransfer::reconstruct_in_parent(
    const MeshBase& mesh,
    size_t parent_elem,
    const std::vector<double>& field) const {
  (void)mesh;
  (void)parent_elem;
  return field;
}

HighOrderTransfer::HighOrderTransfer(const Config& config)
    : config_(config) {}

TransferStats HighOrderTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  // Until a full polynomial reconstruction is implemented, fall back to linear
  // interpolation for MeshFields-based transfer.
  LinearInterpolationTransfer::Config cfg;
  LinearInterpolationTransfer linear(cfg);
  return linear.transfer(old_mesh, new_mesh, old_fields, new_fields, parent_child, options);
}

void HighOrderTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)parent_child;
  LinearInterpolationTransfer::Config cfg;
  LinearInterpolationTransfer linear(cfg);
  linear.prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

void HighOrderTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

std::vector<double> HighOrderTransfer::build_polynomial(
    const MeshBase& mesh,
    size_t elem_id,
    const std::vector<double>& field) const {
  (void)mesh;
  (void)elem_id;
  (void)field;
  return {};
}

double HighOrderTransfer::evaluate_polynomial(
    const std::vector<double>& coefficients,
    const std::array<double, 3>& point) const {
  (void)coefficients;
  (void)point;
  return 0.0;
}

void HighOrderTransfer::apply_limiter(
    std::vector<double>& gradients,
    const MeshBase& mesh,
    size_t elem_id) const {
  (void)gradients;
  (void)mesh;
  (void)elem_id;
}

TransferStats InjectionTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  (void)old_fields;
  (void)new_fields;

  auto start = std::chrono::high_resolution_clock::now();
  MeshBase& writable_new_mesh = const_cast<MeshBase&>(new_mesh);

  TransferStats stats;

  const std::vector<EntityKind> kinds = {EntityKind::Vertex, EntityKind::Volume};
  for (EntityKind kind : kinds) {
    for (const auto& name : old_mesh.field_names(kind)) {
      if (!should_transfer_field_name(options, name)) continue;

      const FieldHandle old_h = MeshFields::get_field_handle(old_mesh, kind, name);
      if (old_h.id == 0) continue;

      const auto type = MeshFields::field_type(old_mesh, old_h);
      const size_t components = MeshFields::field_components(old_mesh, old_h);
      const size_t custom_bpc =
          (type == FieldScalarType::Custom)
              ? old_mesh.field_bytes_per_component_by_name(kind, name)
              : 0u;

      FieldHandle new_h = MeshFields::attach_field(writable_new_mesh, kind, name, type, components, custom_bpc);

      if (kind == EntityKind::Vertex) {
        switch (type) {
          case FieldScalarType::Float64:
            transfer_vertex_numeric_by_weights_or_gid<double>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/false);
            break;
          case FieldScalarType::Float32:
            transfer_vertex_numeric_by_weights_or_gid<float>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/false);
            break;
          case FieldScalarType::Int32:
            transfer_vertex_numeric_by_weights_or_gid<int32_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/true);
            break;
          case FieldScalarType::Int64:
            transfer_vertex_numeric_by_weights_or_gid<int64_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/true);
            break;
          case FieldScalarType::UInt8:
            transfer_vertex_numeric_by_weights_or_gid<uint8_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*round_integral=*/true);
            break;
          case FieldScalarType::Custom:
            transfer_vertex_custom_by_gid(old_mesh, writable_new_mesh, old_h, new_h);
            break;
        }
      } else if (kind == EntityKind::Volume) {
        switch (type) {
          case FieldScalarType::Float64:
            transfer_cell_numeric<double>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            break;
          case FieldScalarType::Float32:
            transfer_cell_numeric<float>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            break;
          case FieldScalarType::Int32:
            transfer_cell_numeric<int32_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            break;
          case FieldScalarType::Int64:
            transfer_cell_numeric<int64_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            break;
          case FieldScalarType::UInt8:
            transfer_cell_numeric<uint8_t>(old_mesh, writable_new_mesh, old_h, new_h, parent_child, /*average_for_coarsening=*/false);
            break;
          case FieldScalarType::Custom:
            transfer_cell_custom(old_mesh, writable_new_mesh, old_h, new_h, parent_child);
            break;
        }
      }

      stats.num_fields++;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  stats.transfer_time = std::chrono::duration<double>(end - start).count();
  return stats;
}

void InjectionTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)parent_child;
  if (new_field.empty()) {
    new_field = old_field;
    return;
  }
  const size_t n = std::min(old_field.size(), new_field.size());
  for (size_t i = 0; i < n; ++i) {
    new_field[i] = old_field[i];
  }
}

void InjectionTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create(const AdaptivityOptions& options) {
  // Prefer the high-level selection when explicitly set (tests use this).
  if (options.field_transfer != FieldTransferType::LINEAR_INTERPOLATION) {
    switch (options.field_transfer) {
      case FieldTransferType::INJECTION:
        return create_injection();
      case FieldTransferType::HIGH_ORDER:
        return create_high_order();
      case FieldTransferType::CONSERVATIVE:
        return create_conservative();
      case FieldTransferType::LINEAR_INTERPOLATION:
      default:
        break;
    }
  }

  // Backward compatibility: select based on legacy prolongation choice.
  switch (options.prolongation_method) {
    case AdaptivityOptions::ProlongationMethod::COPY:
      return create_injection();
    case AdaptivityOptions::ProlongationMethod::HIGH_ORDER_INTERP:
      return create_high_order();
    case AdaptivityOptions::ProlongationMethod::CONSERVATIVE:
      return create_conservative();
    case AdaptivityOptions::ProlongationMethod::LINEAR_INTERP:
    default:
      return create_linear();
  }
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_linear(
    const LinearInterpolationTransfer::Config& config) {
  return std::make_unique<LinearInterpolationTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_conservative(
    const ConservativeTransfer::Config& config) {
  return std::make_unique<ConservativeTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_high_order(
    const HighOrderTransfer::Config& config) {
  return std::make_unique<HighOrderTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_injection() {
  return std::make_unique<InjectionTransfer>();
}

ParentChildMap FieldTransferUtils::build_parent_child_map(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<MarkType>& marks) {
  (void)marks;
  ParentChildMap map;

  const size_t old_cells = old_mesh.n_cells();
  const size_t new_cells = new_mesh.n_cells();
  if (old_cells == 0 || new_cells == 0) return map;

  // Best-effort heuristic: assume ordering is preserved and refined cells
  // contribute a fixed number of children.
  const size_t children_per_parent = std::max<size_t>(1, new_cells / old_cells);

  map.child_to_parent.resize(new_cells);
  for (size_t c = 0; c < new_cells; ++c) {
    const size_t parent = std::min(old_cells - 1, c / children_per_parent);
    map.child_to_parent[c] = parent;
    map.parent_to_children[parent].push_back(c);
  }

  // Trivial vertex injection map by matching GIDs when available.
  for (size_t v = 0; v < new_mesh.n_vertices(); ++v) {
    const gid_t g = new_mesh.vertex_gids()[v];
    const index_t ov = old_mesh.global_to_local_vertex(g);
    if (ov != INVALID_INDEX) {
      map.child_vertex_weights[v] = {{static_cast<size_t>(ov), 1.0}};
    }
  }

  return map;
}

double FieldTransferUtils::check_conservation(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    const std::vector<double>& new_field) {
  (void)old_mesh;
  (void)new_mesh;
  const double old_sum = std::accumulate(old_field.begin(), old_field.end(), 0.0);
  const double new_sum = std::accumulate(new_field.begin(), new_field.end(), 0.0);
  return std::abs(new_sum - old_sum);
}

double FieldTransferUtils::compute_interpolation_error(
    const MeshBase& mesh,
    const std::vector<double>& exact_field,
    const std::vector<double>& interpolated_field) {
  (void)mesh;
  const size_t n = std::min(exact_field.size(), interpolated_field.size());
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::abs(interpolated_field[i] - exact_field[i]));
  }
  return max_err;
}

void FieldTransferUtils::project_field(
    const MeshBase& source_mesh,
    const MeshBase& target_mesh,
    const std::vector<double>& source_field,
    std::vector<double>& target_field) {
  if (target_field.empty()) return;
  if (source_field.empty() || source_mesh.n_vertices() == 0) {
    std::fill(target_field.begin(), target_field.end(), 0.0);
    return;
  }

  // Preserve constant fields exactly.
  if (is_near_constant(source_field, 1e-14)) {
    std::fill(target_field.begin(), target_field.end(), source_field.front());
    return;
  }

  // Best-effort geometric projection: nearest-source-vertex (O(n*m), fine for unit tests).
  const size_t src_n = std::min(source_field.size(), source_mesh.n_vertices());
  for (size_t tv = 0; tv < target_field.size(); ++tv) {
    const auto tp = target_mesh.get_vertex_coords(static_cast<index_t>(tv));
    size_t best = 0;
    long double best_d2 = std::numeric_limits<long double>::infinity();
    for (size_t sv = 0; sv < src_n; ++sv) {
      const auto sp = source_mesh.get_vertex_coords(static_cast<index_t>(sv));
      const long double dx = static_cast<long double>(tp[0]) - static_cast<long double>(sp[0]);
      const long double dy = static_cast<long double>(tp[1]) - static_cast<long double>(sp[1]);
      const long double dz = static_cast<long double>(tp[2]) - static_cast<long double>(sp[2]);
      const long double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = sv;
      }
    }
    target_field[tv] = source_field[best];
  }
}

TransferStats FieldTransferUtils::transfer_all_fields(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) {
  auto transfer = FieldTransferFactory::create(options);
  if (!transfer) {
    throw std::runtime_error("FieldTransferUtils::transfer_all_fields: failed to create transfer strategy");
  }
  return transfer->transfer(old_mesh, new_mesh, old_fields, new_fields, parent_child, options);
}

} // namespace svmp
