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

#include "MeshTopology.h"
#include "../Core/MeshBase.h"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>

namespace svmp {

// Hash function for std::pair<index_t, index_t>
struct PairHash {
  std::size_t operator()(const std::pair<index_t, index_t>& p) const {
    // Cantor pairing function
    return static_cast<std::size_t>(p.first + p.second) * (p.first + p.second + 1) / 2 + p.second;
  }
};

// ---- Adjacency construction ----

void MeshTopology::build_node2cell(const MeshBase& mesh,
                                  std::vector<offset_t>& node2cell_offsets,
                                  std::vector<index_t>& node2cell) {
  size_t n_nodes = mesh.n_nodes();
  size_t n_cells = mesh.n_cells();

  // Count cells per node
  std::vector<index_t> counts(n_nodes, 0);
  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_node] = mesh.cell_nodes_span(static_cast<index_t>(c));
    for (size_t i = 0; i < n_node; ++i) {
      counts[nodes_ptr[i]]++;
    }
  }

  // Build offsets
  node2cell_offsets.resize(n_nodes + 1);
  node2cell_offsets[0] = 0;
  for (size_t n = 0; n < n_nodes; ++n) {
    node2cell_offsets[n + 1] = node2cell_offsets[n] + counts[n];
  }

  // Fill connectivity
  node2cell.resize(static_cast<size_t>(node2cell_offsets.back()));
  std::vector<offset_t> pos(n_nodes, 0);
  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_node] = mesh.cell_nodes_span(static_cast<index_t>(c));
    for (size_t i = 0; i < n_node; ++i) {
      index_t node_id = nodes_ptr[i];
      offset_t offset = node2cell_offsets[node_id];
      node2cell[static_cast<size_t>(offset + pos[node_id]++)] = static_cast<index_t>(c);
    }
  }
}

void MeshTopology::build_node2face(const MeshBase& mesh,
                                  std::vector<offset_t>& node2face_offsets,
                                  std::vector<index_t>& node2face) {
  size_t n_nodes = mesh.n_nodes();
  size_t n_faces = mesh.n_faces();

  if (n_faces == 0) {
    node2face_offsets.clear();
    node2face.clear();
    return;
  }

  // Count faces per node
  std::vector<index_t> counts(n_nodes, 0);
  for (size_t f = 0; f < n_faces; ++f) {
    auto [nodes_ptr, n_node] = mesh.face_nodes_span(static_cast<index_t>(f));
    for (size_t i = 0; i < n_node; ++i) {
      counts[nodes_ptr[i]]++;
    }
  }

  // Build offsets
  node2face_offsets.resize(n_nodes + 1);
  node2face_offsets[0] = 0;
  for (size_t n = 0; n < n_nodes; ++n) {
    node2face_offsets[n + 1] = node2face_offsets[n] + counts[n];
  }

  // Fill connectivity
  node2face.resize(static_cast<size_t>(node2face_offsets.back()));
  std::vector<offset_t> pos(n_nodes, 0);
  for (size_t f = 0; f < n_faces; ++f) {
    auto [nodes_ptr, n_node] = mesh.face_nodes_span(static_cast<index_t>(f));
    for (size_t i = 0; i < n_node; ++i) {
      index_t node_id = nodes_ptr[i];
      offset_t offset = node2face_offsets[node_id];
      node2face[static_cast<size_t>(offset + pos[node_id]++)] = static_cast<index_t>(f);
    }
  }
}

void MeshTopology::build_cell2cell(const MeshBase& mesh,
                                  std::vector<offset_t>& cell2cell_offsets,
                                  std::vector<index_t>& cell2cell) {
  size_t n_cells = mesh.n_cells();
  size_t n_faces = mesh.n_faces();

  if (n_faces > 0) {
    // Build via faces (faster and more accurate)
    std::vector<std::unordered_set<index_t>> neighbors(n_cells);

    for (size_t f = 0; f < n_faces; ++f) {
      const auto& fc = mesh.face_cells(static_cast<index_t>(f));
      if (fc[0] >= 0 && fc[1] >= 0) {
        neighbors[fc[0]].insert(fc[1]);
        neighbors[fc[1]].insert(fc[0]);
      }
    }

    // Convert to CSR
    cell2cell_offsets.resize(n_cells + 1);
    cell2cell_offsets[0] = 0;
    for (size_t c = 0; c < n_cells; ++c) {
      cell2cell_offsets[c + 1] = cell2cell_offsets[c] + static_cast<offset_t>(neighbors[c].size());
    }

    cell2cell.resize(static_cast<size_t>(cell2cell_offsets.back()));
    for (size_t c = 0; c < n_cells; ++c) {
      offset_t offset = cell2cell_offsets[c];
      index_t pos = 0;
      for (index_t neighbor : neighbors[c]) {
        cell2cell[static_cast<size_t>(offset + pos++)] = neighbor;
      }
    }
  } else {
    // Build via shared nodes (slower, less accurate)
    std::vector<offset_t> node2cell_offsets;
    std::vector<index_t> node2cell;
    build_node2cell(mesh, node2cell_offsets, node2cell);

    std::vector<std::unordered_set<index_t>> neighbors(n_cells);

    for (size_t c = 0; c < n_cells; ++c) {
      auto [nodes_ptr, n_node] = mesh.cell_nodes_span(static_cast<index_t>(c));
      for (size_t i = 0; i < n_node; ++i) {
        index_t node_id = nodes_ptr[i];
        offset_t start = node2cell_offsets[node_id];
        offset_t end = node2cell_offsets[node_id + 1];
        for (offset_t j = start; j < end; ++j) {
          index_t other_cell = node2cell[static_cast<size_t>(j)];
          if (other_cell != static_cast<index_t>(c)) {
            neighbors[c].insert(other_cell);
          }
        }
      }
    }

    // Convert to CSR
    cell2cell_offsets.resize(n_cells + 1);
    cell2cell_offsets[0] = 0;
    for (size_t c = 0; c < n_cells; ++c) {
      cell2cell_offsets[c + 1] = cell2cell_offsets[c] + static_cast<offset_t>(neighbors[c].size());
    }

    cell2cell.resize(static_cast<size_t>(cell2cell_offsets.back()));
    for (size_t c = 0; c < n_cells; ++c) {
      offset_t offset = cell2cell_offsets[c];
      index_t pos = 0;
      for (index_t neighbor : neighbors[c]) {
        cell2cell[static_cast<size_t>(offset + pos++)] = neighbor;
      }
    }
  }
}

// ---- Neighbor queries ----

std::vector<index_t> MeshTopology::cell_neighbors(const MeshBase& mesh, index_t cell,
                                                 const std::vector<offset_t>& cell2cell_offsets,
                                                 const std::vector<index_t>& cell2cell) {
  if (!cell2cell_offsets.empty()) {
    // Use provided adjacency
    if (cell < 0 || static_cast<size_t>(cell) >= cell2cell_offsets.size() - 1) {
      return {};
    }
    offset_t start = cell2cell_offsets[cell];
    offset_t end = cell2cell_offsets[cell + 1];
    return std::vector<index_t>(cell2cell.begin() + start, cell2cell.begin() + end);
  } else {
    // Build on demand
    std::vector<offset_t> offsets;
    std::vector<index_t> neighbors;
    build_cell2cell(mesh, offsets, neighbors);
    if (cell < 0 || static_cast<size_t>(cell) >= offsets.size() - 1) {
      return {};
    }
    offset_t start = offsets[cell];
    offset_t end = offsets[cell + 1];
    return std::vector<index_t>(neighbors.begin() + start, neighbors.begin() + end);
  }
}

std::vector<index_t> MeshTopology::node_cells(const MeshBase& mesh, index_t node,
                                             const std::vector<offset_t>& node2cell_offsets,
                                             const std::vector<index_t>& node2cell) {
  if (!node2cell_offsets.empty()) {
    // Use provided adjacency
    if (node < 0 || static_cast<size_t>(node) >= node2cell_offsets.size() - 1) {
      return {};
    }
    offset_t start = node2cell_offsets[node];
    offset_t end = node2cell_offsets[node + 1];
    return std::vector<index_t>(node2cell.begin() + start, node2cell.begin() + end);
  } else {
    // Build on demand
    std::vector<offset_t> offsets;
    std::vector<index_t> cells;
    build_node2cell(mesh, offsets, cells);
    if (node < 0 || static_cast<size_t>(node) >= offsets.size() - 1) {
      return {};
    }
    offset_t start = offsets[node];
    offset_t end = offsets[node + 1];
    return std::vector<index_t>(cells.begin() + start, cells.begin() + end);
  }
}

std::vector<index_t> MeshTopology::face_cells(const MeshBase& mesh, index_t face) {
  if (face < 0 || static_cast<size_t>(face) >= mesh.n_faces()) {
    return {};
  }
  std::vector<index_t> result;
  const auto& fc = mesh.face_cells(face);
  if (fc[0] >= 0) result.push_back(fc[0]);
  if (fc[1] >= 0) result.push_back(fc[1]);
  return result;
}

// ---- Boundary identification ----

std::vector<index_t> MeshTopology::boundary_faces(const MeshBase& mesh) {
  std::vector<index_t> result;
  size_t n_faces = mesh.n_faces();

  for (size_t f = 0; f < n_faces; ++f) {
    const auto& fc = mesh.face_cells(static_cast<index_t>(f));
    if (fc[1] < 0) { // Outer cell is -1 for boundary
      result.push_back(static_cast<index_t>(f));
    }
  }

  return result;
}

std::vector<index_t> MeshTopology::boundary_cells(const MeshBase& mesh) {
  std::unordered_set<index_t> boundary_set;
  size_t n_faces = mesh.n_faces();

  for (size_t f = 0; f < n_faces; ++f) {
    const auto& fc = mesh.face_cells(static_cast<index_t>(f));
    if (fc[1] < 0 && fc[0] >= 0) {
      boundary_set.insert(fc[0]);
    }
  }

  return std::vector<index_t>(boundary_set.begin(), boundary_set.end());
}

std::vector<index_t> MeshTopology::boundary_nodes(const MeshBase& mesh) {
  std::unordered_set<index_t> boundary_set;
  auto bfaces = boundary_faces(mesh);

  for (index_t f : bfaces) {
    auto [nodes_ptr, n_nodes] = mesh.face_nodes_span(f);
    for (size_t i = 0; i < n_nodes; ++i) {
      boundary_set.insert(nodes_ptr[i]);
    }
  }

  return std::vector<index_t>(boundary_set.begin(), boundary_set.end());
}

bool MeshTopology::is_boundary_face(const MeshBase& mesh, index_t face) {
  if (face < 0 || static_cast<size_t>(face) >= mesh.n_faces()) {
    return false;
  }
  const auto& fc = mesh.face_cells(face);
  return fc[1] < 0;
}

// ---- Connectivity analysis ----

std::vector<index_t> MeshTopology::find_components(const MeshBase& mesh) {
  size_t n_cells = mesh.n_cells();
  std::vector<index_t> component_id(n_cells, -1);
  index_t current_component = 0;

  // Build cell-to-cell adjacency
  std::vector<offset_t> cell2cell_offsets;
  std::vector<index_t> cell2cell;
  build_cell2cell(mesh, cell2cell_offsets, cell2cell);

  // BFS to find components
  for (size_t seed = 0; seed < n_cells; ++seed) {
    if (component_id[seed] >= 0) continue; // Already visited

    // Start new component
    std::queue<index_t> queue;
    queue.push(static_cast<index_t>(seed));
    component_id[seed] = current_component;

    while (!queue.empty()) {
      index_t cell = queue.front();
      queue.pop();

      // Visit neighbors
      offset_t start = cell2cell_offsets[cell];
      offset_t end = cell2cell_offsets[cell + 1];
      for (offset_t i = start; i < end; ++i) {
        index_t neighbor = cell2cell[static_cast<size_t>(i)];
        if (component_id[neighbor] < 0) {
          component_id[neighbor] = current_component;
          queue.push(neighbor);
        }
      }
    }

    current_component++;
  }

  return component_id;
}

index_t MeshTopology::count_components(const MeshBase& mesh) {
  auto components = find_components(mesh);
  if (components.empty()) return 0;
  return *std::max_element(components.begin(), components.end()) + 1;
}

bool MeshTopology::is_connected(const MeshBase& mesh) {
  return count_components(mesh) == 1;
}

std::vector<index_t> MeshTopology::cells_within_distance(const MeshBase& mesh,
                                                        index_t cell,
                                                        index_t distance) {
  if (distance < 0) return {};
  if (distance == 0) return {cell};

  // Build cell-to-cell adjacency
  std::vector<offset_t> cell2cell_offsets;
  std::vector<index_t> cell2cell;
  build_cell2cell(mesh, cell2cell_offsets, cell2cell);

  // BFS with distance tracking
  std::unordered_map<index_t, index_t> cell_distance;
  std::queue<index_t> queue;
  queue.push(cell);
  cell_distance[cell] = 0;

  while (!queue.empty()) {
    index_t current = queue.front();
    queue.pop();
    index_t current_dist = cell_distance[current];

    if (current_dist < distance) {
      // Visit neighbors
      offset_t start = cell2cell_offsets[current];
      offset_t end = cell2cell_offsets[current + 1];
      for (offset_t i = start; i < end; ++i) {
        index_t neighbor = cell2cell[static_cast<size_t>(i)];
        if (cell_distance.find(neighbor) == cell_distance.end()) {
          cell_distance[neighbor] = current_dist + 1;
          queue.push(neighbor);
        }
      }
    }
  }

  // Collect cells
  std::vector<index_t> result;
  for (const auto& [c, d] : cell_distance) {
    if (d <= distance) {
      result.push_back(c);
    }
  }

  return result;
}

// ---- Edge operations ----

std::vector<std::array<index_t,2>> MeshTopology::extract_edges(const MeshBase& mesh) {
  std::unordered_set<std::pair<index_t,index_t>, PairHash> edge_set;

  // Extract edges from cells
  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(static_cast<index_t>(c));
    const auto& shape = mesh.cell_shape(static_cast<index_t>(c));

    // Get edges based on cell type
    std::vector<std::pair<int,int>> local_edges;

    switch (shape.family) {
      case CellFamily::Line:
        local_edges = {{0,1}};
        break;
      case CellFamily::Triangle:
        local_edges = {{0,1}, {1,2}, {2,0}};
        break;
      case CellFamily::Quad:
        local_edges = {{0,1}, {1,2}, {2,3}, {3,0}};
        break;
      case CellFamily::Tetra:
        local_edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
        break;
      case CellFamily::Hex:
        local_edges = {{0,1}, {1,2}, {2,3}, {3,0},  // bottom
                      {4,5}, {5,6}, {6,7}, {7,4},  // top
                      {0,4}, {1,5}, {2,6}, {3,7}}; // vertical
        break;
      case CellFamily::Wedge:
        local_edges = {{0,1}, {1,2}, {2,0},  // bottom triangle
                      {3,4}, {4,5}, {5,3},  // top triangle
                      {0,3}, {1,4}, {2,5}}; // vertical
        break;
      case CellFamily::Pyramid:
        local_edges = {{0,1}, {1,2}, {2,3}, {3,0},  // base
                      {0,4}, {1,4}, {2,4}, {3,4}}; // to apex
        break;
      default:
        // For polygons/polyhedra, connect consecutive nodes
        for (size_t i = 0; i < n_nodes; ++i) {
          local_edges.push_back({static_cast<int>(i),
                                static_cast<int>((i + 1) % n_nodes)});
        }
    }

    // Add edges to set (with canonical ordering)
    for (const auto& [i, j] : local_edges) {
      index_t n1 = nodes_ptr[i];
      index_t n2 = nodes_ptr[j];
      if (n1 > n2) std::swap(n1, n2);
      edge_set.insert({n1, n2});
    }
  }

  // Convert to vector
  std::vector<std::array<index_t,2>> edges;
  for (const auto& [n1, n2] : edge_set) {
    edges.push_back({{n1, n2}});
  }

  return edges;
}

// ---- Manifold checks ----

bool MeshTopology::is_manifold(const MeshBase& mesh) {
  // Check that each edge is shared by at most 2 faces
  auto edges = extract_edges(mesh);
  std::unordered_map<std::pair<index_t,index_t>, index_t, PairHash> edge_face_count;

  // Count faces per edge
  for (size_t f = 0; f < mesh.n_faces(); ++f) {
    auto [nodes_ptr, n_nodes] = mesh.face_nodes_span(static_cast<index_t>(f));

    // Extract edges from face
    for (size_t i = 0; i < n_nodes; ++i) {
      index_t n1 = nodes_ptr[i];
      index_t n2 = nodes_ptr[(i + 1) % n_nodes];
      if (n1 > n2) std::swap(n1, n2);
      edge_face_count[{n1, n2}]++;
    }
  }

  // Check manifold condition
  for (const auto& [edge, count] : edge_face_count) {
    if (count > 2) return false;
  }

  // Also check that cells around each node form a valid neighborhood
  // (more complex check omitted for brevity)

  return true;
}

std::vector<std::array<index_t,2>> MeshTopology::non_manifold_edges(const MeshBase& mesh) {
  std::unordered_map<std::pair<index_t,index_t>, index_t, PairHash> edge_face_count;

  // Count faces per edge
  for (size_t f = 0; f < mesh.n_faces(); ++f) {
    auto [nodes_ptr, n_nodes] = mesh.face_nodes_span(static_cast<index_t>(f));

    for (size_t i = 0; i < n_nodes; ++i) {
      index_t n1 = nodes_ptr[i];
      index_t n2 = nodes_ptr[(i + 1) % n_nodes];
      if (n1 > n2) std::swap(n1, n2);
      edge_face_count[{n1, n2}]++;
    }
  }

  // Collect non-manifold edges
  std::vector<std::array<index_t,2>> result;
  for (const auto& [edge, count] : edge_face_count) {
    if (count > 2) {
      result.push_back({{edge.first, edge.second}});
    }
  }

  return result;
}

} // namespace svmp