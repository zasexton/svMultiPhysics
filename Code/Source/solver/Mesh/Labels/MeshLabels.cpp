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
#include <algorithm>
#include <unordered_set>
#include <numeric>
#include <stdexcept>

namespace svmp {

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
    if (label != INVALID_LABEL) {  // Only count actual boundary faces
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
  // This would need to be implemented in MeshBase
  // For now, we can only add to sets, not remove
  throw std::runtime_error("Set removal not yet implemented in MeshBase");
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
  // This would need to be implemented in MeshBase
  throw std::runtime_error("Set removal not yet implemented in MeshBase");
}

std::vector<std::string> MeshLabels::list_sets(const MeshBase& mesh,
                                              EntityKind kind) {
  // This would need MeshBase to expose its set names
  // For now, return empty
  return {};
}

void MeshLabels::create_set_from_label(MeshBase& mesh, EntityKind kind,
                                      const std::string& set_name, label_t label) {
  if (kind == EntityKind::Cell) {
    auto cells = cells_with_region(mesh, label);
    add_to_set(mesh, kind, set_name, cells);
  } else if (kind == EntityKind::Face) {
    auto faces = faces_with_boundary(mesh, label);
    add_to_set(mesh, kind, set_name, faces);
  } else {
    throw std::runtime_error("create_set_from_label only supports Cell and Face entities");
  }
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
  // This would need MeshBase to expose its label registry
  // For now, return empty
  return {};
}

void MeshLabels::clear_label_registry(MeshBase& mesh) {
  // This would need to be implemented in MeshBase
  throw std::runtime_error("Label registry clearing not yet implemented in MeshBase");
}

// ---- Operations ----

std::unordered_map<label_t, label_t> MeshLabels::renumber_labels(MeshBase& mesh,
                                                                EntityKind kind) {
  std::unordered_map<label_t, label_t> old_to_new;

  if (kind == EntityKind::Cell) {
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
  }

  return old_to_new;
}

void MeshLabels::merge_labels(MeshBase& mesh, EntityKind kind,
                             const std::vector<label_t>& source_labels,
                             label_t target_label) {
  std::unordered_set<label_t> sources(source_labels.begin(), source_labels.end());

  if (kind == EntityKind::Cell) {
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
  }
}

std::unordered_map<index_t, label_t> MeshLabels::split_by_connectivity(MeshBase& mesh,
                                                                       EntityKind kind,
                                                                       label_t label) {
  std::unordered_map<index_t, label_t> entity_to_component;

  if (kind == EntityKind::Cell) {
    // Get all cells with this label
    auto cells = cells_with_region(mesh, label);
    std::unordered_set<index_t> cell_set(cells.begin(), cells.end());
    std::unordered_set<index_t> visited;
    label_t component_id = 0;

    // BFS to find connected components
    for (index_t start_cell : cells) {
      if (visited.count(start_cell) > 0) continue;

      std::vector<index_t> queue = {start_cell};
      visited.insert(start_cell);
      entity_to_component[start_cell] = component_id;

      size_t idx = 0;
      while (idx < queue.size()) {
        index_t cell = queue[idx++];

        // Get neighbors by iterating through all faces
        // (MeshBase doesn't provide cell_faces, only face_cells)
        size_t n_faces = mesh.n_faces();
        for (size_t f = 0; f < n_faces; ++f) {
          auto face_cells = mesh.face_cells(static_cast<index_t>(f));

          // Check if this face is connected to our current cell
          bool connected_to_cell = false;
          index_t neighbor = INVALID_INDEX;

          if (face_cells[0] == cell) {
            connected_to_cell = true;
            neighbor = face_cells[1];
          } else if (face_cells[1] == cell) {
            connected_to_cell = true;
            neighbor = face_cells[0];
          }

          if (connected_to_cell && neighbor != INVALID_INDEX &&
              cell_set.count(neighbor) > 0 &&
              visited.count(neighbor) == 0) {
            visited.insert(neighbor);
            queue.push_back(neighbor);
            entity_to_component[neighbor] = component_id;
          }
        }
      }
      component_id++;
    }

    // Apply new component labels
    for (const auto& [entity, comp] : entity_to_component) {
      mesh.set_region_label(entity, label + comp);
    }
  }

  // Similar logic could be implemented for faces

  return entity_to_component;
}

void MeshLabels::copy_labels(const MeshBase& source, MeshBase& target,
                            EntityKind kind) {
  if (kind == EntityKind::Cell) {
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
  }
}

std::vector<label_t> MeshLabels::export_labels(const MeshBase& mesh, EntityKind kind) {
  std::vector<label_t> labels;

  if (kind == EntityKind::Cell) {
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
  }

  return labels;
}

void MeshLabels::import_labels(MeshBase& mesh, EntityKind kind,
                              const std::vector<label_t>& labels) {
  if (kind == EntityKind::Cell) {
    size_t n_cells = std::min(labels.size(), mesh.n_cells());
    for (size_t c = 0; c < n_cells; ++c) {
      mesh.set_region_label(static_cast<index_t>(c), labels[c]);
    }
  } else if (kind == EntityKind::Face) {
    size_t n_faces = std::min(labels.size(), mesh.n_faces());
    for (size_t f = 0; f < n_faces; ++f) {
      mesh.set_boundary_label(static_cast<index_t>(f), labels[f]);
    }
  }
}

} // namespace svmp