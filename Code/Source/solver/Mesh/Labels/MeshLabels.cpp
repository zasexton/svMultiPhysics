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

#include "MeshLabels.h"
#include "../Core/MeshBase.h"
#include "../Topology/MeshTopology.h"
#include <algorithm>
#include <unordered_set>
#include <numeric>
#include <stdexcept>
#include <cstdint>

namespace svmp {

namespace {

struct PairHash {
  std::size_t operator()(const std::pair<index_t, index_t>& p) const noexcept {
    const uint64_t a = static_cast<uint32_t>(p.first);
    const uint64_t b = static_cast<uint32_t>(p.second);
    return static_cast<std::size_t>((a << 32) ^ b);
  }
};

std::pair<index_t, index_t> make_edge(index_t a, index_t b) {
  if (a > b) std::swap(a, b);
  return {a, b};
}

void rebuild_label_registry(MeshBase& mesh,
                            const std::unordered_map<label_t, std::string>& label_to_name) {
  mesh.clear_label_registry();
  for (const auto& kv : label_to_name) {
    mesh.register_label(kv.second, kv.first);
  }
}

std::unordered_map<label_t, std::string> remap_label_registry(
    const std::unordered_map<label_t, std::string>& before,
    const std::unordered_map<label_t, label_t>& old_to_new) {
  std::unordered_map<label_t, std::string> after;
  after.reserve(before.size());

  // First apply remapped labels (take precedence over any pre-existing name at the new ID).
  for (const auto& kv : old_to_new) {
    const auto it = before.find(kv.first);
    if (it != before.end() && !it->second.empty()) {
      after[kv.second] = it->second;
    }
  }

  // Preserve remaining label names that were not remapped and do not conflict.
  for (const auto& kv : before) {
    if (old_to_new.find(kv.first) != old_to_new.end()) {
      continue;
    }
    if (after.find(kv.first) != after.end()) {
      continue;
    }
    after[kv.first] = kv.second;
  }

  return after;
}

label_t max_named_label(const std::unordered_map<label_t, std::string>& registry) {
  label_t max_label = INVALID_LABEL;
  for (const auto& kv : registry) {
    max_label = std::max(max_label, kv.first);
  }
  return max_label;
}

std::string make_unique_name(const std::unordered_set<std::string>& used,
                             const std::string& base) {
  if (used.find(base) == used.end()) {
    return base;
  }

  for (int i = 2; i < 1000000; ++i) {
    const std::string candidate = base + "_" + std::to_string(i);
    if (used.find(candidate) == used.end()) {
      return candidate;
    }
  }

  throw std::runtime_error("Failed to generate unique label name");
}

void register_split_component_names(
    MeshBase& mesh,
    label_t original_label,
    const std::string& base_name,
    const std::unordered_map<label_t, std::string>& registry_before,
    const std::unordered_map<index_t, label_t>& entity_to_new_label) {
  if (base_name.empty()) {
    return;
  }

  std::unordered_set<label_t> new_labels_set;
  new_labels_set.reserve(entity_to_new_label.size());
  for (const auto& kv : entity_to_new_label) {
    if (kv.second != original_label) {
      new_labels_set.insert(kv.second);
    }
  }
  if (new_labels_set.empty()) {
    return;
  }

  std::vector<label_t> new_labels(new_labels_set.begin(), new_labels_set.end());
  std::sort(new_labels.begin(), new_labels.end());

  std::unordered_set<std::string> used_names;
  used_names.reserve(registry_before.size() + new_labels.size());
  for (const auto& kv : registry_before) {
    used_names.insert(kv.second);
  }

  int component_idx = 1;
  for (label_t new_label : new_labels) {
    const std::string candidate = base_name + "_component_" + std::to_string(component_idx++);
    const std::string unique = make_unique_name(used_names, candidate);
    used_names.insert(unique);
    mesh.register_label(unique, new_label);
  }
}

} // namespace

// ---- Region labels ----

void MeshLabels::set_region_label(MeshBase& mesh, index_t cell, label_t label) {
  mesh.set_region_label(cell, label);
}

label_t MeshLabels::region_label(const MeshBase& mesh, index_t cell) {
  return mesh.region_label(cell);
}

std::vector<index_t> MeshLabels::cells_with_region(const MeshBase& mesh, label_t label) {
  return mesh.cells_with_label(label);
}

void MeshLabels::set_region_labels(MeshBase& mesh,
                                  const std::vector<index_t>& cells,
                                  label_t label) {
  for (index_t cell : cells) {
    mesh.set_region_label(cell, label);
  }
}

std::unordered_set<label_t> MeshLabels::unique_region_labels(const MeshBase& mesh) {
  std::unordered_set<label_t> labels;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    labels.insert(mesh.region_label(static_cast<index_t>(c)));
  }

  return labels;
}

std::unordered_map<label_t, size_t> MeshLabels::count_by_region(const MeshBase& mesh) {
  std::unordered_map<label_t, size_t> counts;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    label_t label = mesh.region_label(static_cast<index_t>(c));
    counts[label]++;
  }

  return counts;
}

// ---- Boundary labels ----

void MeshLabels::set_boundary_label(MeshBase& mesh, index_t face, label_t label) {
  mesh.set_boundary_label(face, label);
}

label_t MeshLabels::boundary_label(const MeshBase& mesh, index_t face) {
  return mesh.boundary_label(face);
}

std::vector<index_t> MeshLabels::faces_with_boundary(const MeshBase& mesh, label_t label) {
  return mesh.faces_with_label(label);
}

void MeshLabels::set_boundary_labels(MeshBase& mesh,
                                    const std::vector<index_t>& faces,
                                    label_t label) {
  for (index_t face : faces) {
    mesh.set_boundary_label(face, label);
  }
}

std::unordered_set<label_t> MeshLabels::unique_boundary_labels(const MeshBase& mesh) {
  std::unordered_set<label_t> labels;
  size_t n_faces = mesh.n_faces();

    for (size_t f = 0; f < n_faces; ++f) {
      label_t label = mesh.boundary_label(static_cast<index_t>(f));
      if (label != INVALID_LABEL) {  // Only count labeled faces
        labels.insert(label);
      }
    }

  return labels;
}

std::unordered_map<label_t, size_t> MeshLabels::count_by_boundary(const MeshBase& mesh) {
  std::unordered_map<label_t, size_t> counts;
  size_t n_faces = mesh.n_faces();

  for (size_t f = 0; f < n_faces; ++f) {
    label_t label = mesh.boundary_label(static_cast<index_t>(f));
    if (label != INVALID_LABEL) {
      counts[label]++;
    }
  }

  return counts;
}

// ---- Edge labels ----

void MeshLabels::set_edge_label(MeshBase& mesh, index_t edge, label_t label) {
  mesh.set_edge_label(edge, label);
}

label_t MeshLabels::edge_label(const MeshBase& mesh, index_t edge) {
  return mesh.edge_label(edge);
}

std::vector<index_t> MeshLabels::edges_with_label(const MeshBase& mesh, label_t label) {
  return mesh.edges_with_label(label);
}

void MeshLabels::set_edge_labels(MeshBase& mesh,
                                 const std::vector<index_t>& edges,
                                 label_t label) {
  for (index_t edge : edges) {
    mesh.set_edge_label(edge, label);
  }
}

std::unordered_set<label_t> MeshLabels::unique_edge_labels(const MeshBase& mesh) {
  std::unordered_set<label_t> labels;
  const size_t n_edges = mesh.n_edges();
  for (size_t e = 0; e < n_edges; ++e) {
    const label_t label = mesh.edge_label(static_cast<index_t>(e));
    if (label != INVALID_LABEL) {
      labels.insert(label);
    }
  }
  return labels;
}

std::unordered_map<label_t, size_t> MeshLabels::count_by_edge(const MeshBase& mesh) {
  std::unordered_map<label_t, size_t> counts;
  const size_t n_edges = mesh.n_edges();
  for (size_t e = 0; e < n_edges; ++e) {
    const label_t label = mesh.edge_label(static_cast<index_t>(e));
    if (label != INVALID_LABEL) {
      counts[label]++;
    }
  }
  return counts;
}

// ---- Vertex labels ----

void MeshLabels::set_vertex_label(MeshBase& mesh, index_t vertex, label_t label) {
  mesh.set_vertex_label(vertex, label);
}

label_t MeshLabels::vertex_label(const MeshBase& mesh, index_t vertex) {
  return mesh.vertex_label(vertex);
}

std::vector<index_t> MeshLabels::vertices_with_label(const MeshBase& mesh, label_t label) {
  return mesh.vertices_with_label(label);
}

void MeshLabels::set_vertex_labels(MeshBase& mesh,
                                   const std::vector<index_t>& vertices,
                                   label_t label) {
  for (index_t v : vertices) {
    mesh.set_vertex_label(v, label);
  }
}

std::unordered_set<label_t> MeshLabels::unique_vertex_labels(const MeshBase& mesh) {
  std::unordered_set<label_t> labels;
  const size_t n_vertices = mesh.n_vertices();
  for (size_t v = 0; v < n_vertices; ++v) {
    const label_t label = mesh.vertex_label(static_cast<index_t>(v));
    if (label != INVALID_LABEL) {
      labels.insert(label);
    }
  }
  return labels;
}

std::unordered_map<label_t, size_t> MeshLabels::count_by_vertex(const MeshBase& mesh) {
  std::unordered_map<label_t, size_t> counts;
  const size_t n_vertices = mesh.n_vertices();
  for (size_t v = 0; v < n_vertices; ++v) {
    const label_t label = mesh.vertex_label(static_cast<index_t>(v));
    if (label != INVALID_LABEL) {
      counts[label]++;
    }
  }
  return counts;
}

// ---- Named sets ----

void MeshLabels::add_to_set(MeshBase& mesh, EntityKind kind,
                           const std::string& set_name, index_t entity_id) {
  mesh.add_to_set(kind, set_name, entity_id);
}

void MeshLabels::add_to_set(MeshBase& mesh, EntityKind kind,
                           const std::string& set_name,
                           const std::vector<index_t>& entity_ids) {
  for (index_t id : entity_ids) {
    mesh.add_to_set(kind, set_name, id);
  }
}

void MeshLabels::remove_from_set(MeshBase& mesh, EntityKind kind,
                                const std::string& set_name, index_t entity_id) {
  mesh.remove_from_set(kind, set_name, entity_id);
}

std::vector<index_t> MeshLabels::get_set(const MeshBase& mesh,
                                        EntityKind kind,
                                        const std::string& set_name) {
  // Return a copy, not a reference
  return mesh.get_set(kind, set_name);
}

bool MeshLabels::has_set(const MeshBase& mesh, EntityKind kind,
                        const std::string& set_name) {
  return mesh.has_set(kind, set_name);
}

void MeshLabels::remove_set(MeshBase& mesh, EntityKind kind,
                          const std::string& set_name) {
  mesh.remove_set(kind, set_name);
}

std::vector<std::string> MeshLabels::list_sets(const MeshBase& mesh,
                                              EntityKind kind) {
  return mesh.list_sets(kind);
}

void MeshLabels::create_set_from_label(MeshBase& mesh, EntityKind kind,
                                      const std::string& set_name, label_t label) {
  // Overwrite existing set contents (treat as a "set builder", not append).
  mesh.remove_set(kind, set_name);

  if (kind == EntityKind::Volume) {
    add_to_set(mesh, kind, set_name, cells_with_region(mesh, label));
    return;
  }
  if (kind == EntityKind::Face) {
    add_to_set(mesh, kind, set_name, faces_with_boundary(mesh, label));
    return;
  }
  if (kind == EntityKind::Edge) {
    add_to_set(mesh, kind, set_name, edges_with_label(mesh, label));
    return;
  }
  if (kind == EntityKind::Vertex) {
    add_to_set(mesh, kind, set_name, vertices_with_label(mesh, label));
    return;
  }

  throw std::invalid_argument("create_set_from_label: unsupported entity kind");
}

// ---- Label-name registry ----

void MeshLabels::register_label(MeshBase& mesh, const std::string& name, label_t label) {
  mesh.register_label(name, label);
}

std::string MeshLabels::label_name(const MeshBase& mesh, label_t label) {
  return mesh.label_name(label);
}

label_t MeshLabels::label_from_name(const MeshBase& mesh, const std::string& name) {
  return mesh.label_from_name(name);
}

std::unordered_map<label_t, std::string> MeshLabels::list_label_names(const MeshBase& mesh) {
  return mesh.list_label_names();
}

void MeshLabels::clear_label_registry(MeshBase& mesh) {
  mesh.clear_label_registry();
}

// ---- Operations ----

std::unordered_map<label_t, label_t> MeshLabels::renumber_labels(MeshBase& mesh,
                                                                EntityKind kind) {
  const auto registry_before = mesh.list_label_names();
  std::unordered_map<label_t, label_t> old_to_new;

  if (kind == EntityKind::Volume) {
    auto unique = unique_region_labels(mesh);
    std::vector<label_t> sorted_labels(unique.begin(), unique.end());
    std::sort(sorted_labels.begin(), sorted_labels.end());

    label_t new_label = 0;
    for (label_t old_label : sorted_labels) {
      old_to_new[old_label] = new_label++;
    }

    // Apply renumbering
    size_t n_cells = mesh.n_cells();
    for (size_t c = 0; c < n_cells; ++c) {
      label_t old_label = mesh.region_label(static_cast<index_t>(c));
      mesh.set_region_label(static_cast<index_t>(c), old_to_new[old_label]);
    }
  } else if (kind == EntityKind::Face) {
    auto unique = unique_boundary_labels(mesh);
    std::vector<label_t> sorted_labels(unique.begin(), unique.end());
    std::sort(sorted_labels.begin(), sorted_labels.end());

    label_t new_label = 0;
    for (label_t old_label : sorted_labels) {
      old_to_new[old_label] = new_label++;
    }

    // Apply renumbering
    size_t n_faces = mesh.n_faces();
    for (size_t f = 0; f < n_faces; ++f) {
      label_t old_label = mesh.boundary_label(static_cast<index_t>(f));
      if (old_label != INVALID_LABEL) {
        mesh.set_boundary_label(static_cast<index_t>(f), old_to_new[old_label]);
      }
    }
  } else if (kind == EntityKind::Edge) {
    auto unique = unique_edge_labels(mesh);
    std::vector<label_t> sorted_labels(unique.begin(), unique.end());
    std::sort(sorted_labels.begin(), sorted_labels.end());

    label_t new_label = 0;
    for (label_t old_label : sorted_labels) {
      old_to_new[old_label] = new_label++;
    }

    const size_t n_edges = mesh.n_edges();
    for (size_t e = 0; e < n_edges; ++e) {
      const label_t old_label = mesh.edge_label(static_cast<index_t>(e));
      if (old_label != INVALID_LABEL) {
        mesh.set_edge_label(static_cast<index_t>(e), old_to_new[old_label]);
      }
    }
  } else if (kind == EntityKind::Vertex) {
    auto unique = unique_vertex_labels(mesh);
    std::vector<label_t> sorted_labels(unique.begin(), unique.end());
    std::sort(sorted_labels.begin(), sorted_labels.end());

    label_t new_label = 0;
    for (label_t old_label : sorted_labels) {
      old_to_new[old_label] = new_label++;
    }

    const size_t n_vertices = mesh.n_vertices();
    for (size_t v = 0; v < n_vertices; ++v) {
      const label_t old_label = mesh.vertex_label(static_cast<index_t>(v));
      if (old_label != INVALID_LABEL) {
        mesh.set_vertex_label(static_cast<index_t>(v), old_to_new[old_label]);
      }
    }
  } else {
    throw std::invalid_argument("renumber_labels: unsupported entity kind");
  }

  if (!registry_before.empty() && !old_to_new.empty()) {
    rebuild_label_registry(mesh, remap_label_registry(registry_before, old_to_new));
  }

  return old_to_new;
}

void MeshLabels::merge_labels(MeshBase& mesh, EntityKind kind,
                             const std::vector<label_t>& source_labels,
                             label_t target_label) {
  const auto registry_before = mesh.list_label_names();
  std::unordered_set<label_t> sources(source_labels.begin(), source_labels.end());

  if (kind == EntityKind::Volume) {
    size_t n_cells = mesh.n_cells();
    for (size_t c = 0; c < n_cells; ++c) {
      label_t label = mesh.region_label(static_cast<index_t>(c));
      if (sources.count(label) > 0) {
        mesh.set_region_label(static_cast<index_t>(c), target_label);
      }
    }
  } else if (kind == EntityKind::Face) {
    size_t n_faces = mesh.n_faces();
    for (size_t f = 0; f < n_faces; ++f) {
      label_t label = mesh.boundary_label(static_cast<index_t>(f));
      if (sources.count(label) > 0) {
        mesh.set_boundary_label(static_cast<index_t>(f), target_label);
      }
    }
  } else if (kind == EntityKind::Edge) {
    const size_t n_edges = mesh.n_edges();
    for (size_t e = 0; e < n_edges; ++e) {
      const label_t label = mesh.edge_label(static_cast<index_t>(e));
      if (sources.count(label) > 0) {
        mesh.set_edge_label(static_cast<index_t>(e), target_label);
      }
    }
  } else if (kind == EntityKind::Vertex) {
    const size_t n_vertices = mesh.n_vertices();
    for (size_t v = 0; v < n_vertices; ++v) {
      const label_t label = mesh.vertex_label(static_cast<index_t>(v));
      if (sources.count(label) > 0) {
        mesh.set_vertex_label(static_cast<index_t>(v), target_label);
      }
    }
  } else {
    throw std::invalid_argument("merge_labels: unsupported entity kind");
  }

  if (!registry_before.empty() && !sources.empty()) {
    auto registry_after = registry_before;

    std::string target_name;
    auto it_target = registry_before.find(target_label);
    if (it_target != registry_before.end()) {
      target_name = it_target->second;
    } else {
      for (label_t s : source_labels) {
        auto it = registry_before.find(s);
        if (it != registry_before.end() && !it->second.empty()) {
          target_name = it->second;
          break;
        }
      }
    }

    // Remove the source label names (except the target).
    for (label_t s : sources) {
      if (s != target_label) {
        registry_after.erase(s);
      }
    }

    // If the target label had a different name than the chosen one, update it.
    if (!target_name.empty()) {
      registry_after[target_label] = target_name;
    } else {
      registry_after.erase(target_label);
    }

    rebuild_label_registry(mesh, registry_after);
  }
}

std::unordered_map<index_t, label_t> MeshLabels::split_by_connectivity(MeshBase& mesh,
                                                                       EntityKind kind,
                                                                       label_t label) {
  if (label == INVALID_LABEL) {
    throw std::invalid_argument("split_by_connectivity: cannot split INVALID_LABEL");
  }

  const auto registry_before = mesh.list_label_names();
  const std::string base_name = [&]() -> std::string {
    auto it = registry_before.find(label);
    return (it == registry_before.end()) ? std::string() : it->second;
  }();

  std::unordered_map<index_t, label_t> entity_to_new_label;

  if (kind == EntityKind::Volume) {
    if (mesh.n_cells() == 0) {
      return {};
    }

    // Prefer face-based adjacency when possible (MeshTopology falls back to shared-vertex adjacency).
    if (mesh.n_faces() == 0 && mesh.dim() >= 2) {
      mesh.finalize();
    }

    const auto cells = cells_with_region(mesh, label);
    if (cells.empty()) {
      return {};
    }

    const size_t n_cells = mesh.n_cells();
    std::vector<uint8_t> in_target(n_cells, 0);
    for (index_t c : cells) {
      if (c >= 0 && static_cast<size_t>(c) < n_cells) {
        in_target[static_cast<size_t>(c)] = 1;
      }
    }

    std::vector<index_t> sorted_cells = cells;
    std::sort(sorted_cells.begin(), sorted_cells.end());

    label_t max_label = 0;
    for (size_t c = 0; c < n_cells; ++c) {
      max_label = std::max(max_label, mesh.region_label(static_cast<index_t>(c)));
    }
    const label_t next_label_start = std::max(max_label, max_named_label(registry_before)) + 1;
    label_t next_label = next_label_start;

    std::vector<uint8_t> visited(n_cells, 0);
    bool first_component = true;

    for (index_t start : sorted_cells) {
      if (start < 0 || static_cast<size_t>(start) >= n_cells) continue;
      if (!in_target[static_cast<size_t>(start)] || visited[static_cast<size_t>(start)]) continue;

      const label_t component_label = first_component ? label : next_label++;
      first_component = false;

      std::vector<index_t> queue = {start};
      visited[static_cast<size_t>(start)] = 1;

      size_t qi = 0;
      while (qi < queue.size()) {
        const index_t cell = queue[qi++];
        mesh.set_region_label(cell, component_label);
        entity_to_new_label[cell] = component_label;

        auto neighbors = mesh.cell_neighbors(cell);
        for (index_t nb : neighbors) {
          if (nb < 0 || static_cast<size_t>(nb) >= n_cells) continue;
          if (!in_target[static_cast<size_t>(nb)] || visited[static_cast<size_t>(nb)]) continue;
          visited[static_cast<size_t>(nb)] = 1;
          queue.push_back(nb);
        }
      }
    }

    register_split_component_names(mesh, label, base_name, registry_before, entity_to_new_label);

    return entity_to_new_label;
  }

  if (kind == EntityKind::Face) {
    const size_t n_faces = mesh.n_faces();
    if (n_faces == 0) {
      return {};
    }

    const auto faces = faces_with_boundary(mesh, label);
    if (faces.empty()) {
      return {};
    }

    std::vector<index_t> sorted_faces = faces;
    std::sort(sorted_faces.begin(), sorted_faces.end());

    label_t max_label = 0;
    for (size_t f = 0; f < n_faces; ++f) {
      const auto l = mesh.boundary_label(static_cast<index_t>(f));
      if (l != INVALID_LABEL) max_label = std::max(max_label, l);
    }
    const label_t next_label_start = std::max(max_label, max_named_label(registry_before)) + 1;
    label_t next_label = next_label_start;

    // Determine the adjacency rule for codim-1 entities:
    // - For line-segment faces (2 vertices), connectivity is through shared vertices.
    // - For polygonal faces (>=3 vertices), connectivity is through shared edges.
    const bool vertex_adjacency = [&]() -> bool {
      for (index_t f : sorted_faces) {
        auto [vptr, nv] = mesh.face_vertices_span(f);
        (void)vptr;
        if (nv != 2) {
          return false;
        }
      }
      return true;
    }();

    std::unordered_map<std::pair<index_t, index_t>, std::vector<index_t>, PairHash> edge2faces;
    std::unordered_map<index_t, std::vector<index_t>> vertex2faces;

    if (vertex_adjacency) {
      vertex2faces.reserve(sorted_faces.size() * 2);
      for (index_t f : sorted_faces) {
        auto [vptr, nv] = mesh.face_vertices_span(f);
        for (size_t i = 0; i < nv; ++i) {
          vertex2faces[vptr[i]].push_back(f);
        }
      }
    } else {
      // Build edge -> faces map using only faces with the target label.
      edge2faces.reserve(sorted_faces.size() * 3);
      for (index_t f : sorted_faces) {
        auto [vptr, nv] = mesh.face_vertices_span(f);
        if (nv < 2) continue;
        for (size_t i = 0; i < nv; ++i) {
          edge2faces[make_edge(vptr[i], vptr[(i + 1) % nv])].push_back(f);
        }
      }
    }

    std::vector<uint8_t> in_target(n_faces, 0);
    for (index_t f : sorted_faces) {
      if (f >= 0 && static_cast<size_t>(f) < n_faces) {
        in_target[static_cast<size_t>(f)] = 1;
      }
    }

    std::vector<uint8_t> visited(n_faces, 0);
    bool first_component = true;

    for (index_t start : sorted_faces) {
      if (start < 0 || static_cast<size_t>(start) >= n_faces) continue;
      if (!in_target[static_cast<size_t>(start)] || visited[static_cast<size_t>(start)]) continue;

      const label_t component_label = first_component ? label : next_label++;
      first_component = false;

      std::vector<index_t> queue = {start};
      visited[static_cast<size_t>(start)] = 1;

      size_t qi = 0;
      while (qi < queue.size()) {
        const index_t face = queue[qi++];
        mesh.set_boundary_label(face, component_label);
        entity_to_new_label[face] = component_label;

        auto [vptr, nv] = mesh.face_vertices_span(face);
        if (nv < 2) continue;
        if (vertex_adjacency) {
          for (size_t i = 0; i < nv; ++i) {
            const index_t v = vptr[i];
            auto it = vertex2faces.find(v);
            if (it == vertex2faces.end()) continue;
            for (index_t nb : it->second) {
              if (nb < 0 || static_cast<size_t>(nb) >= n_faces) continue;
              if (!in_target[static_cast<size_t>(nb)] || visited[static_cast<size_t>(nb)]) continue;
              visited[static_cast<size_t>(nb)] = 1;
              queue.push_back(nb);
            }
          }
        } else {
          for (size_t i = 0; i < nv; ++i) {
            const auto edge = make_edge(vptr[i], vptr[(i + 1) % nv]);
            auto it = edge2faces.find(edge);
            if (it == edge2faces.end()) continue;
            for (index_t nb : it->second) {
              if (nb < 0 || static_cast<size_t>(nb) >= n_faces) continue;
              if (!in_target[static_cast<size_t>(nb)] || visited[static_cast<size_t>(nb)]) continue;
              visited[static_cast<size_t>(nb)] = 1;
              queue.push_back(nb);
            }
          }
        }
      }
    }

    register_split_component_names(mesh, label, base_name, registry_before, entity_to_new_label);

    return entity_to_new_label;
  }

  if (kind == EntityKind::Edge) {
    if (mesh.n_edges() == 0 && mesh.dim() >= 2 && mesh.n_cells() > 0) {
      mesh.finalize();
    }
    const size_t n_edges = mesh.n_edges();
    if (n_edges == 0) {
      return {};
    }

    const auto edges = edges_with_label(mesh, label);
    if (edges.empty()) {
      return {};
    }

    std::vector<index_t> sorted_edges = edges;
    std::sort(sorted_edges.begin(), sorted_edges.end());

    std::vector<uint8_t> in_target(n_edges, 0);
    for (index_t e : sorted_edges) {
      if (e >= 0 && static_cast<size_t>(e) < n_edges) {
        in_target[static_cast<size_t>(e)] = 1;
      }
    }

    label_t max_label = 0;
    for (size_t e = 0; e < n_edges; ++e) {
      const auto l = mesh.edge_label(static_cast<index_t>(e));
      if (l != INVALID_LABEL) max_label = std::max(max_label, l);
    }
    const label_t next_label_start = std::max(max_label, max_named_label(registry_before)) + 1;
    label_t next_label = next_label_start;

    // Build vertex -> edges adjacency restricted to the target edges.
    std::unordered_map<index_t, std::vector<index_t>> vertex2edges;
    vertex2edges.reserve(sorted_edges.size() * 2);
    for (index_t e : sorted_edges) {
      const auto ev = mesh.edge_vertices(e);
      vertex2edges[ev[0]].push_back(e);
      vertex2edges[ev[1]].push_back(e);
    }

    std::vector<uint8_t> visited(n_edges, 0);
    bool first_component = true;

    for (index_t start : sorted_edges) {
      if (start < 0 || static_cast<size_t>(start) >= n_edges) continue;
      if (!in_target[static_cast<size_t>(start)] || visited[static_cast<size_t>(start)]) continue;

      const label_t component_label = first_component ? label : next_label++;
      first_component = false;

      std::vector<index_t> queue = {start};
      visited[static_cast<size_t>(start)] = 1;

      size_t qi = 0;
      while (qi < queue.size()) {
        const index_t edge = queue[qi++];
        mesh.set_edge_label(edge, component_label);
        entity_to_new_label[edge] = component_label;

        const auto ev = mesh.edge_vertices(edge);
        for (index_t v : {ev[0], ev[1]}) {
          auto it = vertex2edges.find(v);
          if (it == vertex2edges.end()) continue;
          for (index_t nb : it->second) {
            if (nb < 0 || static_cast<size_t>(nb) >= n_edges) continue;
            if (!in_target[static_cast<size_t>(nb)] || visited[static_cast<size_t>(nb)]) continue;
            visited[static_cast<size_t>(nb)] = 1;
            queue.push_back(nb);
          }
        }
      }
    }

    register_split_component_names(mesh, label, base_name, registry_before, entity_to_new_label);
    return entity_to_new_label;
  }

  if (kind == EntityKind::Vertex) {
    const size_t n_vertices = mesh.n_vertices();
    if (n_vertices == 0) {
      return {};
    }

    const auto vertices = vertices_with_label(mesh, label);
    if (vertices.empty()) {
      return {};
    }

    std::vector<index_t> sorted_vertices = vertices;
    std::sort(sorted_vertices.begin(), sorted_vertices.end());

    std::vector<uint8_t> in_target(n_vertices, 0);
    for (index_t v : sorted_vertices) {
      if (v >= 0 && static_cast<size_t>(v) < n_vertices) {
        in_target[static_cast<size_t>(v)] = 1;
      }
    }

    label_t max_label = 0;
    for (size_t v = 0; v < n_vertices; ++v) {
      const auto l = mesh.vertex_label(static_cast<index_t>(v));
      if (l != INVALID_LABEL) max_label = std::max(max_label, l);
    }
    const label_t next_label_start = std::max(max_label, max_named_label(registry_before)) + 1;
    label_t next_label = next_label_start;

    // Build adjacency among target-labeled vertices via mesh edges (or extracted edges if missing).
    const auto& mesh_edges = mesh.edge2vertex();
    std::vector<std::array<index_t,2>> extracted_edges;
    const auto& edges = [&]() -> const std::vector<std::array<index_t,2>>& {
      if (!mesh_edges.empty()) {
        return mesh_edges;
      }
      extracted_edges = MeshTopology::extract_edges(mesh);
      return extracted_edges;
    }();

    std::unordered_map<index_t, std::vector<index_t>> vertex_neighbors;
    vertex_neighbors.reserve(sorted_vertices.size());
    for (const auto& ev : edges) {
      const index_t a = ev[0];
      const index_t b = ev[1];
      if (a < 0 || b < 0) continue;
      if (static_cast<size_t>(a) >= n_vertices || static_cast<size_t>(b) >= n_vertices) continue;
      if (!in_target[static_cast<size_t>(a)] || !in_target[static_cast<size_t>(b)]) continue;
      vertex_neighbors[a].push_back(b);
      vertex_neighbors[b].push_back(a);
    }

    std::vector<uint8_t> visited(n_vertices, 0);
    bool first_component = true;

    for (index_t start : sorted_vertices) {
      if (start < 0 || static_cast<size_t>(start) >= n_vertices) continue;
      if (!in_target[static_cast<size_t>(start)] || visited[static_cast<size_t>(start)]) continue;

      const label_t component_label = first_component ? label : next_label++;
      first_component = false;

      std::vector<index_t> queue = {start};
      visited[static_cast<size_t>(start)] = 1;

      size_t qi = 0;
      while (qi < queue.size()) {
        const index_t v = queue[qi++];
        mesh.set_vertex_label(v, component_label);
        entity_to_new_label[v] = component_label;

        auto it = vertex_neighbors.find(v);
        if (it == vertex_neighbors.end()) continue;
        for (index_t nb : it->second) {
          if (nb < 0 || static_cast<size_t>(nb) >= n_vertices) continue;
          if (!in_target[static_cast<size_t>(nb)] || visited[static_cast<size_t>(nb)]) continue;
          visited[static_cast<size_t>(nb)] = 1;
          queue.push_back(nb);
        }
      }
    }

    register_split_component_names(mesh, label, base_name, registry_before, entity_to_new_label);
    return entity_to_new_label;
  }

  throw std::invalid_argument("split_by_connectivity: unsupported entity kind");
}

void MeshLabels::copy_labels(const MeshBase& source, MeshBase& target,
                            EntityKind kind) {
  if (kind == EntityKind::Volume) {
    size_t n_cells = std::min(source.n_cells(), target.n_cells());
    for (size_t c = 0; c < n_cells; ++c) {
      label_t label = source.region_label(static_cast<index_t>(c));
      target.set_region_label(static_cast<index_t>(c), label);
    }
  } else if (kind == EntityKind::Face) {
    size_t n_faces = std::min(source.n_faces(), target.n_faces());
    for (size_t f = 0; f < n_faces; ++f) {
      label_t label = source.boundary_label(static_cast<index_t>(f));
      target.set_boundary_label(static_cast<index_t>(f), label);
    }
  } else if (kind == EntityKind::Edge) {
    size_t n_edges = std::min(source.n_edges(), target.n_edges());
    for (size_t e = 0; e < n_edges; ++e) {
      label_t label = source.edge_label(static_cast<index_t>(e));
      target.set_edge_label(static_cast<index_t>(e), label);
    }
  } else if (kind == EntityKind::Vertex) {
    size_t n_vertices = std::min(source.n_vertices(), target.n_vertices());
    for (size_t v = 0; v < n_vertices; ++v) {
      label_t label = source.vertex_label(static_cast<index_t>(v));
      target.set_vertex_label(static_cast<index_t>(v), label);
    }
  } else {
    throw std::invalid_argument("copy_labels: unsupported entity kind");
  }
}

std::vector<label_t> MeshLabels::export_labels(const MeshBase& mesh, EntityKind kind) {
  std::vector<label_t> labels;

  if (kind == EntityKind::Volume) {
    size_t n_cells = mesh.n_cells();
    labels.resize(n_cells);
    for (size_t c = 0; c < n_cells; ++c) {
      labels[c] = mesh.region_label(static_cast<index_t>(c));
    }
  } else if (kind == EntityKind::Face) {
    size_t n_faces = mesh.n_faces();
    labels.resize(n_faces);
    for (size_t f = 0; f < n_faces; ++f) {
      labels[f] = mesh.boundary_label(static_cast<index_t>(f));
    }
  } else if (kind == EntityKind::Edge) {
    size_t n_edges = mesh.n_edges();
    labels.resize(n_edges);
    for (size_t e = 0; e < n_edges; ++e) {
      labels[e] = mesh.edge_label(static_cast<index_t>(e));
    }
  } else if (kind == EntityKind::Vertex) {
    size_t n_vertices = mesh.n_vertices();
    labels.resize(n_vertices);
    for (size_t v = 0; v < n_vertices; ++v) {
      labels[v] = mesh.vertex_label(static_cast<index_t>(v));
    }
  } else {
    throw std::invalid_argument("export_labels: unsupported entity kind");
  }

  return labels;
}

void MeshLabels::import_labels(MeshBase& mesh, EntityKind kind,
                              const std::vector<label_t>& labels) {
  if (kind == EntityKind::Volume) {
    size_t n_cells = std::min(labels.size(), mesh.n_cells());
    for (size_t c = 0; c < n_cells; ++c) {
      mesh.set_region_label(static_cast<index_t>(c), labels[c]);
    }
  } else if (kind == EntityKind::Face) {
    size_t n_faces = std::min(labels.size(), mesh.n_faces());
    for (size_t f = 0; f < n_faces; ++f) {
      mesh.set_boundary_label(static_cast<index_t>(f), labels[f]);
    }
  } else if (kind == EntityKind::Edge) {
    size_t n_edges = std::min(labels.size(), mesh.n_edges());
    for (size_t e = 0; e < n_edges; ++e) {
      mesh.set_edge_label(static_cast<index_t>(e), labels[e]);
    }
  } else if (kind == EntityKind::Vertex) {
    size_t n_vertices = std::min(labels.size(), mesh.n_vertices());
    for (size_t v = 0; v < n_vertices; ++v) {
      mesh.set_vertex_label(static_cast<index_t>(v), labels[v]);
    }
  } else {
    throw std::invalid_argument("import_labels: unsupported entity kind");
  }
}

} // namespace svmp
