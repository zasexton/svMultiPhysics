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

#include "HangingVertexConstraints.h"
#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"
#include "../Fields/MeshFields.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {

namespace {

struct ConstraintWire {
  int64_t ints[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  double weights[4] = {0.0, 0.0, 0.0, 0.0};
};

static_assert(sizeof(ConstraintWire) == 96, "ConstraintWire size must be 96 bytes");

bool wires_match(const ConstraintWire& a, const ConstraintWire& b, real_t tol) {
  for (int i = 0; i < 8; ++i) {
    if (a.ints[i] != b.ints[i]) {
      return false;
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (std::abs(a.weights[i] - b.weights[i]) > tol) {
      return false;
    }
  }
  return true;
}

} // namespace

// Simplified helper functions to get topological edges/facets based on cell family.
// IMPORTANT: MeshBase::dim() is the spatial embedding dimension (often 3),
// not the topological dimension (2 for surface meshes embedded in 3D).
static std::vector<std::pair<size_t, size_t>> get_cell_edges_by_family(CellFamily family,
                                                                       size_t n_verts) {
  std::vector<std::pair<size_t, size_t>> edges;

  switch (family) {
    case CellFamily::Line:
      edges = {{0, 1}};
      break;
    case CellFamily::Triangle:
      edges = {{0, 1}, {1, 2}, {2, 0}};
      break;
    case CellFamily::Quad:
      edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
      break;
    case CellFamily::Tetra:
      edges = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
      break;
    case CellFamily::Hex:
      edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4},
               {0, 4}, {1, 5}, {2, 6}, {3, 7}};
      break;
    case CellFamily::Wedge:
      edges = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3},
               {0, 3}, {1, 4}, {2, 5}};
      break;
    case CellFamily::Pyramid:
      edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};
      break;
    case CellFamily::Polygon:
      for (size_t i = 0; i < n_verts; ++i) {
        edges.emplace_back(i, (i + 1) % n_verts);
      }
      break;
    default:
      break;
  }

  if (edges.empty() && n_verts >= 2) {
    for (size_t i = 0; i < n_verts; ++i) {
      edges.emplace_back(i, (i + 1) % n_verts);
    }
  }

  return edges;
}

// Co-dimension-1 facets (edges in 2D, faces in 3D) by cell family.
static std::vector<std::vector<size_t>> get_cell_facets_by_family(CellFamily family,
                                                                  size_t n_verts) {
  std::vector<std::vector<size_t>> faces;

  switch (family) {
    case CellFamily::Triangle:
      faces = {{0, 1}, {1, 2}, {2, 0}};
      break;
    case CellFamily::Quad:
      faces = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
      break;
    case CellFamily::Tetra:
      faces = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
      break;
    case CellFamily::Hex:
      faces = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
      break;
    case CellFamily::Wedge:
      faces = {{0, 1, 2}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
      break;
    case CellFamily::Pyramid:
      faces = {{0, 1, 2, 3}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
      break;
    case CellFamily::Polygon:
      for (size_t i = 0; i < n_verts; ++i) {
        faces.push_back({i, (i + 1) % n_verts});
      }
      break;
    default:
      break;
  }

  return faces;
}

static int cell_topological_dim(CellFamily family) {
  switch (family) {
    case CellFamily::Line:
      return 1;
    case CellFamily::Triangle:
    case CellFamily::Quad:
    case CellFamily::Polygon:
      return 2;
    case CellFamily::Tetra:
    case CellFamily::Hex:
    case CellFamily::Wedge:
    case CellFamily::Pyramid:
    case CellFamily::Polyhedron:
      return 3;
    default:
      return 0;
  }
}

static int mesh_topological_dim(const MeshBase& mesh) {
  int tdim = 0;
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    tdim = std::max(tdim, cell_topological_dim(mesh.cell_shape(c).family));
  }
  return tdim;
}

// Constructor
HangingVertexConstraints::HangingVertexConstraints() {
  // Initialize empty
}

// Destructor
HangingVertexConstraints::~HangingVertexConstraints() {
  // Clean up automatically handled by STL containers
}

// Main detection method
void HangingVertexConstraints::detect_hanging_vertices(
    const MeshBase& mesh,
    const std::vector<size_t>* refinement_levels) {

  // Clear existing constraints
  clear();

  // Detect edge hanging vertices (2D and 3D)
  detect_edge_hanging(mesh, refinement_levels);

  // For topologically 3D meshes, also detect face hanging vertices
  if (mesh_topological_dim(mesh) == 3) {
    detect_face_hanging(mesh, refinement_levels);
  }

  // Update internal maps
  update_maps();
}

void HangingVertexConstraints::detect_hanging_vertices(
    const DistributedMesh& mesh,
    const std::vector<size_t>* refinement_levels) {
  detect_hanging_vertices(mesh.local_mesh(), refinement_levels);
  synchronize(mesh);
}

bool HangingVertexConstraints::synchronize(const DistributedMesh& mesh, real_t weight_tolerance) {
  if (mesh.world_size() <= 1) {
    return true;
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL) {
    return false;
  }

  MPI_Comm comm = mesh.mpi_comm();
  int my_rank = 0;
  int world = 1;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &world);

  const auto& vertex_gids = mesh.local_mesh().vertex_gids();

  std::vector<std::vector<ConstraintWire>> send_by_owner(static_cast<size_t>(world));
  bool local_ok = true;

  for (const auto& kv : constraints_) {
    const HangingVertexConstraint& c = kv.second;
    if (!c.is_valid()) {
      local_ok = false;
      continue;
    }

    const index_t local_constrained = c.constrained_vertex;
    if (local_constrained < 0 || static_cast<size_t>(local_constrained) >= vertex_gids.size()) {
      local_ok = false;
      continue;
    }

    const rank_t owner = mesh.owner_rank_vertex(local_constrained);
    if (owner < 0 || owner >= world) {
      local_ok = false;
      continue;
    }

    ConstraintWire wire;
    wire.ints[0] = static_cast<int64_t>(vertex_gids[static_cast<size_t>(local_constrained)]);
    wire.ints[1] = static_cast<int64_t>(
        c.parent_type == ConstraintParentType::Edge ? 0 : (c.parent_type == ConstraintParentType::Face ? 1 : -1));

    const size_t n_parents = c.parent_vertices.size();
    if (n_parents < 2 || n_parents > 4 || c.weights.size() != n_parents) {
      local_ok = false;
      continue;
    }
    wire.ints[2] = static_cast<int64_t>(n_parents);
    wire.ints[7] = static_cast<int64_t>(c.refinement_level);

    std::vector<std::pair<gid_t, real_t>> parents;
    parents.reserve(n_parents);
    for (size_t i = 0; i < n_parents; ++i) {
      const index_t pv = c.parent_vertices[i];
      if (pv < 0 || static_cast<size_t>(pv) >= vertex_gids.size()) {
        local_ok = false;
        parents.clear();
        break;
      }
      parents.emplace_back(vertex_gids[static_cast<size_t>(pv)], c.weights[i]);
    }
    if (parents.empty()) {
      continue;
    }

    std::sort(parents.begin(), parents.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t i = 0; i < parents.size(); ++i) {
      wire.ints[3 + static_cast<int>(i)] = static_cast<int64_t>(parents[i].first);
      wire.weights[i] = static_cast<double>(parents[i].second);
    }
    for (size_t i = parents.size(); i < 4; ++i) {
      wire.ints[3 + static_cast<int>(i)] = static_cast<int64_t>(INVALID_GID);
      wire.weights[i] = 0.0;
    }

    send_by_owner[static_cast<size_t>(owner)].push_back(wire);
  }

  // Send proposed constraints to the owner rank of each constrained vertex.
  std::vector<int> send_counts(world, 0);
  for (int r = 0; r < world; ++r) {
    send_counts[r] = static_cast<int>(send_by_owner[static_cast<size_t>(r)].size() * sizeof(ConstraintWire));
  }

  std::vector<int> recv_counts(world, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

  std::vector<int> send_displs(world + 1, 0);
  std::vector<int> recv_displs(world + 1, 0);
  for (int r = 0; r < world; ++r) {
    send_displs[r + 1] = send_displs[r] + send_counts[r];
    recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
  }

  std::vector<char> send_buf(static_cast<size_t>(send_displs[world]));
  for (int r = 0; r < world; ++r) {
    const auto& vec = send_by_owner[static_cast<size_t>(r)];
    if (vec.empty()) {
      continue;
    }
    const size_t bytes = vec.size() * sizeof(ConstraintWire);
    std::memcpy(send_buf.data() + send_displs[r], vec.data(), bytes);
  }

  std::vector<char> recv_buf(static_cast<size_t>(recv_displs[world]));
  MPI_Alltoallv(send_buf.data(),
                send_counts.data(),
                send_displs.data(),
                MPI_BYTE,
                recv_buf.data(),
                recv_counts.data(),
                recv_displs.data(),
                MPI_BYTE,
                comm);

  struct Proposal {
    ConstraintWire wire;
    int src_rank = -1;
  };

  std::unordered_map<gid_t, std::vector<Proposal>> proposals;
  proposals.reserve(static_cast<size_t>(recv_displs[world] / static_cast<int>(sizeof(ConstraintWire)) + 8));

  for (int src = 0; src < world; ++src) {
    const int begin = recv_displs[src];
    const int end = recv_displs[src + 1];
    if (begin == end) {
      continue;
    }
    for (int off = begin; off + static_cast<int>(sizeof(ConstraintWire)) <= end;
         off += static_cast<int>(sizeof(ConstraintWire))) {
      ConstraintWire w;
      std::memcpy(&w, recv_buf.data() + off, sizeof(ConstraintWire));
      const gid_t constrained_gid = static_cast<gid_t>(w.ints[0]);
      proposals[constrained_gid].push_back(Proposal{w, src});
    }
  }

  // Owner ranks select a canonical constraint for their owned constrained vertices.
  std::vector<ConstraintWire> owned_canonical;
  owned_canonical.reserve(proposals.size());
  bool local_conflict = false;

  for (auto& kv : proposals) {
    const gid_t constrained_gid = kv.first;
    const index_t local_v = mesh.global_to_local_vertex(constrained_gid);
    if (local_v == INVALID_INDEX || !mesh.is_owned_vertex(local_v)) {
      continue;
    }

    auto& vec = kv.second;
    if (vec.empty()) {
      continue;
    }

    size_t best = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
      if (vec[i].src_rank == my_rank) {
        best = i;
        break;
      }
      if (vec[i].src_rank < vec[best].src_rank) {
        best = i;
      }
    }

    const ConstraintWire& chosen = vec[best].wire;
    for (const auto& p : vec) {
      if (!wires_match(chosen, p.wire, weight_tolerance)) {
        local_conflict = true;
        break;
      }
    }

    owned_canonical.push_back(chosen);
  }

  // Broadcast canonical constraints: all ranks receive all owner-defined constraints,
  // and keep only those for vertices they have locally.
  int local_n = static_cast<int>(owned_canonical.size());
  std::vector<int> n_by_rank(world, 0);
  MPI_Allgather(&local_n, 1, MPI_INT, n_by_rank.data(), 1, MPI_INT, comm);

  std::vector<int> disp(world + 1, 0);
  for (int r = 0; r < world; ++r) {
    disp[r + 1] = disp[r] + n_by_rank[r];
  }

  std::vector<int> n_by_rank_bytes(world, 0);
  std::vector<int> disp_bytes(world, 0);
  for (int r = 0; r < world; ++r) {
    n_by_rank_bytes[r] = n_by_rank[r] * static_cast<int>(sizeof(ConstraintWire));
    disp_bytes[r] = disp[r] * static_cast<int>(sizeof(ConstraintWire));
  }

  std::vector<ConstraintWire> all(static_cast<size_t>(disp[world]));
  MPI_Allgatherv(owned_canonical.data(),
                 local_n * static_cast<int>(sizeof(ConstraintWire)),
                 MPI_BYTE,
                 all.data(),
                 n_by_rank_bytes.data(),
                 disp_bytes.data(),
                 MPI_BYTE,
                 comm);

  std::unordered_map<index_t, HangingVertexConstraint> next;
  next.reserve(all.size());

  bool missing_parents = false;

  for (const auto& w : all) {
    const gid_t constrained_gid = static_cast<gid_t>(w.ints[0]);
    const index_t local_constrained = mesh.global_to_local_vertex(constrained_gid);
    if (local_constrained == INVALID_INDEX) {
      continue;
    }

    const int64_t parent_type_tag = w.ints[1];
    ConstraintParentType parent_type =
        (parent_type_tag == 0) ? ConstraintParentType::Edge
                               : (parent_type_tag == 1) ? ConstraintParentType::Face : ConstraintParentType::Invalid;

    const int64_t n_parents = w.ints[2];
    if (n_parents < 2 || n_parents > 4) {
      missing_parents = true;
      continue;
    }

    std::vector<index_t> parents;
    std::vector<real_t> weights;
    parents.reserve(static_cast<size_t>(n_parents));
    weights.reserve(static_cast<size_t>(n_parents));

    bool have_all = true;
    for (int i = 0; i < n_parents; ++i) {
      const gid_t pgid = static_cast<gid_t>(w.ints[3 + i]);
      const index_t pl = mesh.global_to_local_vertex(pgid);
      if (pl == INVALID_INDEX) {
        have_all = false;
        break;
      }
      parents.push_back(pl);
      weights.push_back(static_cast<real_t>(w.weights[i]));
    }
    if (!have_all) {
      missing_parents = true;
      continue;
    }

    HangingVertexConstraint c;
    c.constrained_vertex = local_constrained;
    c.parent_type = parent_type;
    c.parent_vertices = std::move(parents);
    c.weights = std::move(weights);
    c.refinement_level = static_cast<size_t>(w.ints[7]);

    // Preserve local adjacency info if it exists on this rank.
    auto it_old = constraints_.find(local_constrained);
    if (it_old != constraints_.end()) {
      c.adjacent_cells = it_old->second.adjacent_cells;
    }

    next[local_constrained] = std::move(c);
  }

  constraints_ = std::move(next);
  update_maps();

  const int ok_int = (local_ok && !missing_parents && !local_conflict) ? 1 : 0;
  int global_ok = 1;
  MPI_Allreduce(&ok_int, &global_ok, 1, MPI_INT, MPI_MIN, comm);
  return global_ok == 1;
#else
  (void)mesh;
  (void)weight_tolerance;
  return true;
#endif
}

// Detect hanging vertices on edges
void HangingVertexConstraints::detect_edge_hanging(
    const MeshBase& mesh,
    const std::vector<size_t>* refinement_levels) {

  const size_t num_cells = mesh.n_cells();

  // Build edge to cells map
  std::map<std::pair<index_t, index_t>, std::set<index_t>> edge_to_cells;

  for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
    auto cell_verts = mesh.cell_vertices(cell_id);

    // Get edges for this cell type (topological, independent of spatial embedding)
    auto family = mesh.cell_shape(cell_id).family;
    auto edges = get_cell_edges_by_family(family, cell_verts.size());

    for (const auto& edge : edges) {
      index_t v1 = cell_verts[edge.first];
      index_t v2 = cell_verts[edge.second];
      auto edge_key = make_edge(v1, v2);
      edge_to_cells[edge_key].insert(cell_id);
    }
  }

  // Check each edge for hanging vertices
  for (const auto& [edge, cells] : edge_to_cells) {
    if (cells.size() < 2) continue; // Boundary edge

    index_t v1 = edge.first;
    index_t v2 = edge.second;

    // Find vertices that lie on this edge
    std::set<index_t> edge_vertices;

    for (index_t cell_id : cells) {
      auto cell_verts = mesh.cell_vertices(cell_id);

      // Check all vertices of the cell
      for (index_t v : cell_verts) {
        if (v == v1 || v == v2) continue;

        // Check if vertex lies on the edge
        if (is_edge_midpoint(mesh, v, v1, v2)) {
          edge_vertices.insert(v);
        }
      }
    }

    // Check if any edge vertices are hanging
    for (index_t v : edge_vertices) {
      // Count how many cells contain this vertex
      size_t containing_cells = 0;
      std::set<index_t> adjacent_cells;

      for (index_t cell_id : cells) {
        auto cell_verts = mesh.cell_vertices(cell_id);
        if (std::find(cell_verts.begin(), cell_verts.end(), v) !=
            cell_verts.end()) {
          containing_cells++;
          adjacent_cells.insert(cell_id);
        }
      }

      // If vertex is not in all cells sharing the edge, it's hanging
      if (containing_cells > 0 && containing_cells < cells.size()) {
        HangingVertexConstraint constraint;
        constraint.constrained_vertex = v;
        constraint.parent_type = ConstraintParentType::Edge;
        constraint.parent_vertices = {v1, v2};
        constraint.weights = compute_edge_weights(mesh, v, v1, v2);
        constraint.adjacent_cells = adjacent_cells;

        // Determine refinement level if provided
        if (refinement_levels) {
          size_t max_level = 0;
          for (index_t cell_id : adjacent_cells) {
            max_level = std::max(max_level, (*refinement_levels)[cell_id]);
          }
          constraint.refinement_level = max_level;
        } else {
          constraint.refinement_level = 1;
        }

        add_constraint(constraint);
      }
    }
  }
}

// Detect hanging vertices on faces (3D only)
void HangingVertexConstraints::detect_face_hanging(
    const MeshBase& mesh,
    const std::vector<size_t>* refinement_levels) {

  const size_t num_cells = mesh.n_cells();

  // Build face to cells map
  std::map<std::set<index_t>, std::set<index_t>> face_to_cells;

  for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
    auto cell_verts = mesh.cell_vertices(cell_id);

      // Get faces for this cell type
    auto faces = get_cell_facets_by_family(mesh.cell_shape(cell_id).family, cell_verts.size());

    for (const auto& face : faces) {
      std::set<index_t> face_set(face.begin(), face.end());

      // Convert local to global indices
      std::set<index_t> global_face;
      for (index_t local_id : face_set) {
        global_face.insert(cell_verts[local_id]);
      }

      face_to_cells[global_face].insert(cell_id);
    }
  }

  // Check each face for hanging vertices
  for (const auto& [face, cells] : face_to_cells) {
    if (cells.size() < 2) continue; // Boundary face

    std::vector<index_t> face_vertices(face.begin(), face.end());

    // Find vertices that lie on this face
    std::set<index_t> on_face_vertices;

    for (index_t cell_id : cells) {
      auto cell_verts = mesh.cell_vertices(cell_id);

      // Check all vertices of the cell
      for (index_t v : cell_verts) {
        if (face.count(v) > 0) continue; // Skip face corners

        // Check if vertex lies on the face
        if (is_face_centroid(mesh, v, face_vertices)) {
          on_face_vertices.insert(v);
        }
      }
    }

    // Check if any face vertices are hanging
    for (index_t v : on_face_vertices) {
      // Count how many cells contain this vertex
      size_t containing_cells = 0;
      std::set<index_t> adjacent_cells;

      for (index_t cell_id : cells) {
        auto cell_verts = mesh.cell_vertices(cell_id);
        if (std::find(cell_verts.begin(), cell_verts.end(), v) !=
            cell_verts.end()) {
          containing_cells++;
          adjacent_cells.insert(cell_id);
        }
      }

      // If vertex is not in all cells sharing the face, it's hanging
      if (containing_cells > 0 && containing_cells < cells.size()) {
        HangingVertexConstraint constraint;
        constraint.constrained_vertex = v;
        constraint.parent_type = ConstraintParentType::Face;
        constraint.parent_vertices = face_vertices;
        constraint.weights = compute_face_weights(mesh, v, face_vertices);
        constraint.adjacent_cells = adjacent_cells;

        // Determine refinement level if provided
        if (refinement_levels) {
          size_t max_level = 0;
          for (index_t cell_id : adjacent_cells) {
            max_level = std::max(max_level, (*refinement_levels)[cell_id]);
          }
          constraint.refinement_level = max_level;
        } else {
          constraint.refinement_level = 1;
        }

        add_constraint(constraint);
      }
    }
  }
}

// Add a constraint
bool HangingVertexConstraints::add_constraint(const HangingVertexConstraint& constraint) {
  // Validate constraint
  if (!constraint.is_valid()) {
    return false;
  }

  // Check for duplicate
  if (constraints_.count(constraint.constrained_vertex) > 0) {
    return false;
  }

  // Add to main constraint map
  constraints_[constraint.constrained_vertex] = constraint;

  // Update edge hanging map
  if (constraint.parent_type == ConstraintParentType::Edge) {
    auto edge = make_edge(constraint.parent_vertices[0],
                         constraint.parent_vertices[1]);
    edge_hanging_map_[edge].push_back(constraint.constrained_vertex);
  }

  // Update affected cells
  for (index_t cell : constraint.adjacent_cells) {
    affected_cells_.insert(cell);
  }

  return true;
}

// Remove a constraint
bool HangingVertexConstraints::remove_constraint(index_t vertex_id) {
  auto it = constraints_.find(vertex_id);
  if (it == constraints_.end()) {
    return false;
  }

  const auto& constraint = it->second;

  // Remove from edge hanging map
  if (constraint.parent_type == ConstraintParentType::Edge) {
    auto edge = make_edge(constraint.parent_vertices[0],
                         constraint.parent_vertices[1]);
    auto edge_it = edge_hanging_map_.find(edge);
    if (edge_it != edge_hanging_map_.end()) {
      auto& vec = edge_it->second;
      vec.erase(std::remove(vec.begin(), vec.end(), vertex_id), vec.end());
      if (vec.empty()) {
        edge_hanging_map_.erase(edge_it);
      }
    }
  }

  // Remove from constraints map
  constraints_.erase(it);

  // Rebuild affected_cells_ set
  affected_cells_.clear();
  for (const auto& [vid, c] : constraints_) {
    for (index_t cell : c.adjacent_cells) {
      affected_cells_.insert(cell);
    }
  }

  return true;
}

// Clear all constraints
void HangingVertexConstraints::clear() {
  constraints_.clear();
  edge_hanging_map_.clear();
  affected_cells_.clear();
}

// Check if vertex is hanging
bool HangingVertexConstraints::is_hanging(index_t vertex_id) const {
  return constraints_.find(vertex_id) != constraints_.end();
}

// Get constraint for a vertex
HangingVertexConstraint HangingVertexConstraints::get_constraint(index_t vertex_id) const {
  auto it = constraints_.find(vertex_id);
  if (it == constraints_.end()) {
    return HangingVertexConstraint(); // Return invalid constraint
  }
  return it->second;
}

// Get all hanging vertices
std::vector<index_t> HangingVertexConstraints::get_hanging_vertices() const {
  std::vector<index_t> result;
  result.reserve(constraints_.size());
  for (const auto& [vertex_id, constraint] : constraints_) {
    result.push_back(vertex_id);
  }
  return result;
}

// Get hanging vertices on an edge
std::vector<index_t> HangingVertexConstraints::get_edge_hanging_vertices(
    index_t v1, index_t v2) const {

  auto edge = make_edge(v1, v2);
  auto it = edge_hanging_map_.find(edge);
  if (it == edge_hanging_map_.end()) {
    return {};
  }

  return it->second;
}

// Get hanging vertices on a face
std::vector<index_t> HangingVertexConstraints::get_face_hanging_vertices(
    const std::vector<index_t>& face_vertices) const {

  // Search through all constraints for face hanging vertices
  std::vector<index_t> result;
  std::set<index_t> face_set(face_vertices.begin(), face_vertices.end());

  for (const auto& [vertex_id, constraint] : constraints_) {
    if (constraint.parent_type == ConstraintParentType::Face) {
      std::set<index_t> parent_set(constraint.parent_vertices.begin(),
                                    constraint.parent_vertices.end());
      if (parent_set == face_set) {
        result.push_back(vertex_id);
      }
    }
  }

  return result;
}

// Get constraints by type
std::vector<HangingVertexConstraint> HangingVertexConstraints::get_constraints_by_type(
    ConstraintParentType parent_type) const {
  std::vector<HangingVertexConstraint> result;

  for (const auto& [vertex_id, constraint] : constraints_) {
    if (constraint.parent_type == parent_type) {
      result.push_back(constraint);
    }
  }

  return result;
}

// Check if an edge has hanging vertices
bool HangingVertexConstraints::edge_has_hanging(index_t v1, index_t v2) const {
  auto edge = make_edge(v1, v2);
  auto it = edge_hanging_map_.find(edge);
  return it != edge_hanging_map_.end() && !it->second.empty();
}

// Check if a face has hanging vertices
bool HangingVertexConstraints::face_has_hanging(const std::vector<index_t>& face_vertices) const {
  std::set<index_t> face_set(face_vertices.begin(), face_vertices.end());

  for (const auto& [vertex_id, constraint] : constraints_) {
    if (constraint.parent_type == ConstraintParentType::Face) {
      std::set<index_t> parent_set(constraint.parent_vertices.begin(),
                                    constraint.parent_vertices.end());
      if (parent_set == face_set) {
        return true;
      }
    }
  }

  return false;
}

// Generate constraint matrix
std::map<index_t, std::map<index_t, real_t>>
HangingVertexConstraints::generate_constraint_matrix() const {

  std::map<index_t, std::map<index_t, real_t>> matrix;

  for (const auto& [vertex_id, constraint] : constraints_) {
    std::map<index_t, real_t> row;

    for (size_t i = 0; i < constraint.parent_vertices.size(); ++i) {
      row[constraint.parent_vertices[i]] = constraint.weights[i];
    }

    matrix[vertex_id] = row;
  }

  return matrix;
}

// Apply constraints to solution vector
void HangingVertexConstraints::apply_constraints(
    std::vector<real_t>& solution, size_t num_components) const {

  for (const auto& [vertex_id, constraint] : constraints_) {
    // Apply constraint to each component
    for (size_t comp = 0; comp < num_components; ++comp) {
      real_t value = 0.0;

      for (size_t i = 0; i < constraint.parent_vertices.size(); ++i) {
        index_t parent_idx = constraint.parent_vertices[i] * num_components + comp;
        value += constraint.weights[i] * solution[parent_idx];
      }

      index_t constrained_idx = vertex_id * num_components + comp;
      solution[constrained_idx] = value;
    }
  }
}

// Compute statistics
HangingVertexConstraints::Statistics HangingVertexConstraints::compute_statistics() const {
  Statistics stats;

  stats.num_edge_hanging = 0;
  stats.num_face_hanging = 0;
  stats.max_refinement_level = 0;
  stats.num_affected_cells = affected_cells_.size();

  for (const auto& [vertex_id, constraint] : constraints_) {
    if (constraint.parent_type == ConstraintParentType::Edge) {
      stats.num_edge_hanging++;
    } else if (constraint.parent_type == ConstraintParentType::Face) {
      stats.num_face_hanging++;
    }

    stats.max_refinement_level = std::max(stats.max_refinement_level,
                                         constraint.refinement_level);
  }

  return stats;
}

// Validate constraints
bool HangingVertexConstraints::validate(const MeshBase& mesh, real_t tolerance) const {
  const size_t num_vertices = mesh.n_vertices();

  for (const auto& [vertex_id, constraint] : constraints_) {
    // Check constrained vertex exists
    if (vertex_id >= num_vertices) {
      return false;
    }

    // Check all parent vertices exist
    for (index_t parent : constraint.parent_vertices) {
      if (parent >= num_vertices) {
        return false;
      }
    }

    // Check weights sum to 1.0
    real_t weight_sum = std::accumulate(constraint.weights.begin(),
                                       constraint.weights.end(), real_t(0.0));
    if (std::abs(weight_sum - 1.0) > tolerance) {
      return false;
    }

    // Check no circular dependencies
    for (index_t parent : constraint.parent_vertices) {
      if (is_hanging(parent)) {
        return false;
      }
    }

    // Check parent entity is topologically valid
    if (constraint.parent_type == ConstraintParentType::Edge) {
      if (constraint.parent_vertices.size() != 2) {
        return false;
      }
    } else if (constraint.parent_type == ConstraintParentType::Face) {
      if (constraint.parent_vertices.size() < 3) {
        return false;
      }
    }
  }

  return true;
}

// Export constraints to fields for visualization
void HangingVertexConstraints::export_to_fields(
    MeshBase& mesh,
    const std::string& field_name) const {

  // This would require MeshFields implementation
  // For now, provide a stub that could be implemented when fields are available
  // The actual implementation would create fields showing:
  // - Which vertices are hanging (binary field)
  // - Refinement level at each hanging vertex
  // - Parent type (edge vs face)

  (void)mesh;         // Suppress unused parameter warning
  (void)field_name;   // Suppress unused parameter warning

  // TODO: Implement field export when MeshFields API is available
}

// Update internal maps
void HangingVertexConstraints::update_maps() {
  edge_hanging_map_.clear();
  affected_cells_.clear();

  for (const auto& [vertex_id, constraint] : constraints_) {
    if (constraint.parent_type == ConstraintParentType::Edge) {
      auto edge = make_edge(constraint.parent_vertices[0],
                           constraint.parent_vertices[1]);
      edge_hanging_map_[edge].push_back(vertex_id);
    }

    for (index_t cell : constraint.adjacent_cells) {
      affected_cells_.insert(cell);
    }
  }
}

// Helper: Check if vertex is edge midpoint
bool HangingVertexConstraints::is_edge_midpoint(
    const MeshBase& mesh,
    index_t v,
    index_t v1,
    index_t v2,
    real_t tolerance) const {

  // Get vertex coordinates
  auto p = mesh.get_vertex_coords(v);
  auto p1 = mesh.get_vertex_coords(v1);
  auto p2 = mesh.get_vertex_coords(v2);

  // Compute expected midpoint
  std::array<double, 3> midpoint;
  for (int i = 0; i < 3; ++i) {
    midpoint[i] = 0.5 * (p1[i] + p2[i]);
  }

  // Check distance
  double dist_sq = 0.0;
  for (int i = 0; i < 3; ++i) {
    double diff = p[i] - midpoint[i];
    dist_sq += diff * diff;
  }

  // Tolerance based on edge length
  double edge_len_sq = 0.0;
  for (int i = 0; i < 3; ++i) {
    double diff = p2[i] - p1[i];
    edge_len_sq += diff * diff;
  }

  return dist_sq < tolerance * tolerance * edge_len_sq;
}

// Helper: Check if vertex is face centroid
bool HangingVertexConstraints::is_face_centroid(
    const MeshBase& mesh,
    index_t v,
    const std::vector<index_t>& face_vertices,
    real_t tolerance) const {

  // Get vertex coordinates
  auto p = mesh.get_vertex_coords(v);

  // Compute expected centroid
  std::array<double, 3> centroid = {0.0, 0.0, 0.0};
  for (index_t fv : face_vertices) {
    auto pf = mesh.get_vertex_coords(fv);
    for (int i = 0; i < 3; ++i) {
      centroid[i] += pf[i];
    }
  }

  double n = static_cast<double>(face_vertices.size());
  for (int i = 0; i < 3; ++i) {
    centroid[i] /= n;
  }

  // Check distance
  double dist_sq = 0.0;
  for (int i = 0; i < 3; ++i) {
    double diff = p[i] - centroid[i];
    dist_sq += diff * diff;
  }

  // Tolerance based on face size
  double face_size_sq = 0.0;
  for (size_t j = 0; j < face_vertices.size(); ++j) {
    auto pj = mesh.get_vertex_coords(face_vertices[j]);
    for (int i = 0; i < 3; ++i) {
      double diff = pj[i] - centroid[i];
      face_size_sq += diff * diff;
    }
  }
  face_size_sq /= n;

  return dist_sq < tolerance * tolerance * face_size_sq;
}

// Helper: Compute edge interpolation weights
std::vector<real_t> HangingVertexConstraints::compute_edge_weights(
    const MeshBase& mesh,
    index_t v,
    index_t v1,
    index_t v2) const {

  // For now, assume linear interpolation at midpoint
  // In general, could compute parametric position
  (void)mesh;  // Suppress unused parameter warning
  (void)v;     // Suppress unused parameter warning
  (void)v1;    // Suppress unused parameter warning
  (void)v2;    // Suppress unused parameter warning
  return {0.5, 0.5};
}

// Helper: Compute face interpolation weights
std::vector<real_t> HangingVertexConstraints::compute_face_weights(
    const MeshBase& mesh,
    index_t v,
    const std::vector<index_t>& face_vertices) const {

  // For now, assume equal weights (centroid)
  // In general, could compute barycentric coordinates
  (void)mesh;  // Suppress unused parameter warning
  (void)v;     // Suppress unused parameter warning
  size_t n = face_vertices.size();
  return std::vector<real_t>(n, 1.0 / n);
}

// ============================================================================
// HangingVertexUtils implementation
// ============================================================================

// Check if refinement will create hanging vertices
bool HangingVertexUtils::will_create_hanging(
    const MeshBase& mesh,
    const std::set<index_t>& cells_to_refine) {

  const size_t num_cells = mesh.n_cells();

  // Build edge to cells map
  std::map<std::pair<index_t, index_t>, std::set<index_t>> edge_to_cells;

  for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
    auto cell_verts = mesh.cell_vertices(cell_id);

    // Get edges for this cell type
    auto edges = get_cell_edges_by_family(mesh.cell_shape(cell_id).family, cell_verts.size());

    for (const auto& edge : edges) {
      index_t v1 = cell_verts[edge.first];
      index_t v2 = cell_verts[edge.second];
      index_t v_min = std::min(v1, v2);
      index_t v_max = std::max(v1, v2);
      edge_to_cells[{v_min, v_max}].insert(cell_id);
    }
  }

  // Check each edge
  for (const auto& [edge, cells] : edge_to_cells) {
    if (cells.size() < 2) continue; // Boundary edge

    // Count how many cells sharing this edge will be refined
    size_t refining_count = 0;
    for (index_t cell_id : cells) {
      if (cells_to_refine.count(cell_id) > 0) {
        refining_count++;
      }
    }

    // If some but not all cells will be refined, hanging vertices will be created
    if (refining_count > 0 && refining_count < cells.size()) {
      return true;
    }
  }

  // Similar check for faces in topologically 3D meshes
  if (mesh_topological_dim(mesh) == 3) {
    std::map<std::set<index_t>, std::set<index_t>> face_to_cells;

    for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
      auto cell_verts = mesh.cell_vertices(cell_id);

      // Get faces for this cell type
      auto faces = get_cell_facets_by_family(mesh.cell_shape(cell_id).family, cell_verts.size());

      for (const auto& face : faces) {
        std::set<index_t> face_set;
        for (index_t local_id : face) {
          face_set.insert(cell_verts[local_id]);
        }
        face_to_cells[face_set].insert(cell_id);
      }
    }

    for (const auto& [face, cells] : face_to_cells) {
      if (cells.size() < 2) continue; // Boundary face

      size_t refining_count = 0;
      for (index_t cell_id : cells) {
        if (cells_to_refine.count(cell_id) > 0) {
          refining_count++;
        }
      }

      if (refining_count > 0 && refining_count < cells.size()) {
        return true;
      }
    }
  }

  return false;
}

// Find cells that need closure refinement
std::set<index_t> HangingVertexUtils::find_closure_cells(
    const MeshBase& mesh,
    const HangingVertexConstraints& constraints) {

  std::set<index_t> closure_cells;

  // For each hanging vertex, add cells that don't contain it but share its parent entity
  for (const auto& [vertex_id, constraint] : constraints.get_all_constraints()) {
    // Get all cells that share the parent entity
    std::set<index_t> parent_cells;

    if (constraint.parent_type == ConstraintParentType::Edge) {
      // Find cells containing both edge vertices
      // This is a simplified implementation - would need proper vertex-to-cell connectivity
      const size_t num_cells = mesh.n_cells();

      for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
        auto cell_verts = mesh.cell_vertices(cell_id);

        bool has_v1 = std::find(cell_verts.begin(), cell_verts.end(),
                              constraint.parent_vertices[0]) != cell_verts.end();
        bool has_v2 = std::find(cell_verts.begin(), cell_verts.end(),
                              constraint.parent_vertices[1]) != cell_verts.end();

        if (has_v1 && has_v2) {
          parent_cells.insert(cell_id);
        }
      }
    } else if (constraint.parent_type == ConstraintParentType::Face) {
      // Find cells containing all face vertices
      const size_t num_cells = mesh.n_cells();

      for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
        auto cell_verts = mesh.cell_vertices(cell_id);

        bool has_all = true;
        for (index_t fv : constraint.parent_vertices) {
          if (std::find(cell_verts.begin(), cell_verts.end(), fv) == cell_verts.end()) {
            has_all = false;
            break;
          }
        }

        if (has_all) {
          parent_cells.insert(cell_id);
        }
      }
    }

    // Add cells that don't contain the hanging vertex
    for (index_t cell_id : parent_cells) {
      if (constraint.adjacent_cells.count(cell_id) == 0) {
        closure_cells.insert(cell_id);
      }
    }
  }

  return closure_cells;
}

// Compute maximum level difference
size_t HangingVertexUtils::compute_max_level_difference(
    const MeshBase& mesh,
    const std::vector<size_t>& refinement_levels) {

  size_t max_diff = 0;
  const size_t num_cells = mesh.n_cells();

  // Check all cell pairs that share a face
  for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
    // Get neighbors through faces
    auto cell_verts = mesh.cell_vertices(cell_id);
    auto faces = get_cell_facets_by_family(mesh.cell_shape(cell_id).family, cell_verts.size());

    for (const auto& face : faces) {
      // Find cells sharing this face
      std::set<index_t> face_cells;

      // This is simplified - would need proper face-to-cell connectivity
      for (index_t other_id = 0; other_id < static_cast<index_t>(num_cells); ++other_id) {
        if (other_id == cell_id) continue;

        auto other_verts = mesh.cell_vertices(other_id);

        // Check if other cell shares this face
        size_t shared_count = 0;
        for (size_t local_id : face) {
          index_t v = cell_verts[local_id];
          if (std::find(other_verts.begin(), other_verts.end(), v) != other_verts.end()) {
            shared_count++;
          }
        }

        if (shared_count == face.size()) {
          face_cells.insert(other_id);
        }
      }

      // Compute level differences
      for (index_t other_id : face_cells) {
        size_t level1 = refinement_levels[cell_id];
        size_t level2 = refinement_levels[other_id];
        size_t diff = (level1 > level2) ? (level1 - level2) : (level2 - level1);
        max_diff = std::max(max_diff, diff);
      }
    }
  }

  return max_diff;
}

bool HangingVertexUtils::is_valid_hanging_pattern(const MeshBase& mesh,
                                                 const HangingVertexConstraints& constraints,
                                                 const std::vector<size_t>& refinement_levels) {
  // Check that hanging vertices are properly constrained
  // and that level differences are at most 1
  return validate_balance(mesh, refinement_levels, nullptr);
}

// Private helper methods

bool HangingVertexUtils::vertex_on_edge(const MeshBase& mesh,
                                       index_t vertex,
                                       index_t v1,
                                       index_t v2,
                                       real_t tolerance) {
  auto coords = mesh.get_vertex_coords(vertex);
  auto coords1 = mesh.get_vertex_coords(v1);
  auto coords2 = mesh.get_vertex_coords(v2);

  // Helper lambda to compute distance from point to line segment
  auto point_to_segment_distance = [](const std::array<real_t, 3>& p,
                                      const std::array<real_t, 3>& v1,
                                      const std::array<real_t, 3>& v2) -> real_t {
    std::array<real_t, 3> edge = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    std::array<real_t, 3> to_point = {p[0] - v1[0], p[1] - v1[1], p[2] - v1[2]};

    real_t edge_len_sq = edge[0]*edge[0] + edge[1]*edge[1] + edge[2]*edge[2];
    if (edge_len_sq < 1e-20) {
      // Degenerate edge
      return std::sqrt(to_point[0]*to_point[0] + to_point[1]*to_point[1] + to_point[2]*to_point[2]);
    }

    // Project point onto line
    real_t t = (to_point[0]*edge[0] + to_point[1]*edge[1] + to_point[2]*edge[2]) / edge_len_sq;
    t = std::max(0.0, std::min(1.0, t));  // Clamp to segment

    std::array<real_t, 3> closest = {
      v1[0] + t * edge[0],
      v1[1] + t * edge[1],
      v1[2] + t * edge[2]
    };

    std::array<real_t, 3> diff = {
      p[0] - closest[0],
      p[1] - closest[1],
      p[2] - closest[2]
    };

    return std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
  };

  real_t dist = point_to_segment_distance(coords, coords1, coords2);

  // Also check parametric position on edge
  std::array<real_t, 3> edge = {
    coords2[0] - coords1[0],
    coords2[1] - coords1[1],
    coords2[2] - coords1[2]
  };
  std::array<real_t, 3> to_point = {
    coords[0] - coords1[0],
    coords[1] - coords1[1],
    coords[2] - coords1[2]
  };

  real_t edge_len_sq = edge[0]*edge[0] + edge[1]*edge[1] + edge[2]*edge[2];
  if (edge_len_sq > 1e-20) {
    real_t t = (to_point[0]*edge[0] + to_point[1]*edge[1] + to_point[2]*edge[2]) / edge_len_sq;
    // Check if point is interior to edge (not at endpoints)
    if (t > 0.1 && t < 0.9 && dist < tolerance) {
      return true;
    }
  }

  return false;
}

bool HangingVertexUtils::vertex_on_face(const MeshBase& mesh,
                                       index_t vertex,
                                       const std::vector<index_t>& face_vertices,
                                       real_t tolerance) {
  if (face_vertices.size() < 3) return false;

  // Get coordinates
  auto p = mesh.get_vertex_coords(vertex);
  std::vector<std::array<real_t, 3>> face_coords;
  for (index_t fv : face_vertices) {
    face_coords.push_back(mesh.get_vertex_coords(fv));
  }

  // Compute face normal and check if point is coplanar
  // Using first 3 vertices to define plane
  std::array<real_t, 3> v01 = {
    face_coords[1][0] - face_coords[0][0],
    face_coords[1][1] - face_coords[0][1],
    face_coords[1][2] - face_coords[0][2]
  };
  std::array<real_t, 3> v02 = {
    face_coords[2][0] - face_coords[0][0],
    face_coords[2][1] - face_coords[0][1],
    face_coords[2][2] - face_coords[0][2]
  };

  // Cross product for normal
  std::array<real_t, 3> normal = {
    v01[1]*v02[2] - v01[2]*v02[1],
    v01[2]*v02[0] - v01[0]*v02[2],
    v01[0]*v02[1] - v01[1]*v02[0]
  };

  real_t normal_len = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
  if (normal_len < 1e-20) return false;  // Degenerate face

  normal[0] /= normal_len;
  normal[1] /= normal_len;
  normal[2] /= normal_len;

  // Check distance to plane
  std::array<real_t, 3> to_point = {
    p[0] - face_coords[0][0],
    p[1] - face_coords[0][1],
    p[2] - face_coords[0][2]
  };

  real_t dist_to_plane = std::abs(to_point[0]*normal[0] + to_point[1]*normal[1] + to_point[2]*normal[2]);
  if (dist_to_plane > tolerance) return false;

  // Check if point is inside face polygon (simplified check)
  // This is a basic implementation - production code would use proper
  // point-in-polygon test
  return true;
}

bool HangingVertexUtils::compute_barycentric_weights(const MeshBase& mesh,
                                                    index_t point_id,
                                                    const std::vector<index_t>& face_vertices,
                                                    std::vector<real_t>& weights) {
  weights.clear();

  if (face_vertices.size() < 3) return false;

  auto p = mesh.get_vertex_coords(point_id);

  if (face_vertices.size() == 3) {
    // Triangle - compute standard barycentric coordinates
    auto v0 = mesh.get_vertex_coords(face_vertices[0]);
    auto v1 = mesh.get_vertex_coords(face_vertices[1]);
    auto v2 = mesh.get_vertex_coords(face_vertices[2]);

    // Vectors from v0
    std::array<real_t, 3> v01 = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    std::array<real_t, 3> v02 = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    std::array<real_t, 3> v0p = {p[0]-v0[0], p[1]-v0[1], p[2]-v0[2]};

    // Compute dot products
    real_t d00 = v01[0]*v01[0] + v01[1]*v01[1] + v01[2]*v01[2];
    real_t d01 = v01[0]*v02[0] + v01[1]*v02[1] + v01[2]*v02[2];
    real_t d11 = v02[0]*v02[0] + v02[1]*v02[1] + v02[2]*v02[2];
    real_t d20 = v0p[0]*v01[0] + v0p[1]*v01[1] + v0p[2]*v01[2];
    real_t d21 = v0p[0]*v02[0] + v0p[1]*v02[1] + v0p[2]*v02[2];

    real_t denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-20) return false;

    real_t v = (d11 * d20 - d01 * d21) / denom;
    real_t w = (d00 * d21 - d01 * d20) / denom;
    real_t u = 1.0 - v - w;

    weights.push_back(u);
    weights.push_back(v);
    weights.push_back(w);

    return true;
  } else if (face_vertices.size() == 4) {
    // Quad - use bilinear interpolation
    // Simplified: just use average weights
    weights.resize(4, 0.25);
    return true;
  }

  return false;
}

std::vector<std::pair<index_t, index_t>>
HangingVertexUtils::get_cell_edges(const MeshBase& mesh, index_t cell_id) {
  std::vector<std::pair<index_t, index_t>> edges;
  auto verts = mesh.cell_vertices(cell_id);

  // Get cell type to determine edge connectivity
  // This is simplified - would need proper topology information
  if (verts.size() == 4) {  // Tet
    edges.push_back({verts[0], verts[1]});
    edges.push_back({verts[0], verts[2]});
    edges.push_back({verts[0], verts[3]});
    edges.push_back({verts[1], verts[2]});
    edges.push_back({verts[1], verts[3]});
    edges.push_back({verts[2], verts[3]});
  } else if (verts.size() == 8) {  // Hex
    // Add hex edges
    edges.push_back({verts[0], verts[1]});
    edges.push_back({verts[1], verts[2]});
    edges.push_back({verts[2], verts[3]});
    edges.push_back({verts[3], verts[0]});
    edges.push_back({verts[4], verts[5]});
    edges.push_back({verts[5], verts[6]});
    edges.push_back({verts[6], verts[7]});
    edges.push_back({verts[7], verts[4]});
    edges.push_back({verts[0], verts[4]});
    edges.push_back({verts[1], verts[5]});
    edges.push_back({verts[2], verts[6]});
    edges.push_back({verts[3], verts[7]});
  }

  return edges;
}

std::vector<std::vector<index_t>>
HangingVertexUtils::get_cell_faces(const MeshBase& mesh, index_t cell_id) {
  std::vector<std::vector<index_t>> faces;
  auto verts = mesh.cell_vertices(cell_id);

  // Get cell type to determine face connectivity
  if (verts.size() == 4) {  // Tet
    faces.push_back({verts[0], verts[1], verts[2]});
    faces.push_back({verts[0], verts[1], verts[3]});
    faces.push_back({verts[0], verts[2], verts[3]});
    faces.push_back({verts[1], verts[2], verts[3]});
  } else if (verts.size() == 8) {  // Hex
    faces.push_back({verts[0], verts[1], verts[2], verts[3]});
    faces.push_back({verts[4], verts[5], verts[6], verts[7]});
    faces.push_back({verts[0], verts[1], verts[5], verts[4]});
    faces.push_back({verts[1], verts[2], verts[6], verts[5]});
    faces.push_back({verts[2], verts[3], verts[7], verts[6]});
    faces.push_back({verts[3], verts[0], verts[4], verts[7]});
  }

  return faces;
}

std::unordered_map<index_t, HangingVertexInfo>
HangingVertexUtils::detect_hanging_vertices(const MeshBase& mesh,
                                           const std::unordered_set<index_t>& refined_cells,
                                           const std::vector<size_t>* refinement_levels) {
  std::unordered_map<index_t, HangingVertexInfo> hanging_vertices;

  // For each refined cell, check its vertices
  for (index_t cell_id : refined_cells) {
    auto cell_verts = mesh.cell_vertices(cell_id);

    // Check each vertex to see if it's hanging
    for (index_t v : cell_verts) {
      // Get cells containing this vertex
      auto v_cells = mesh.vertex_cells(v);

      // Check if vertex is on edge/face of coarser neighbor
      for (index_t neighbor_id : v_cells) {
        if (refined_cells.find(neighbor_id) == refined_cells.end()) {
          // This neighbor is coarser

          // Check if vertex lies on neighbor's edges
          auto edges = get_cell_edges(mesh, neighbor_id);
          for (const auto& edge : edges) {
            if (vertex_on_edge(mesh, v, edge.first, edge.second)) {
              HangingVertexInfo info;
              info.vertex_id = v;
              info.parent_type = ConstraintParentType::Edge;
              info.parent_vertices = {edge.first, edge.second};
              info.weights = {0.5, 0.5};  // Midpoint weights
              info.coarse_neighbors.push_back(neighbor_id);
              hanging_vertices[v] = info;
              break;
            }
          }

          // Check if vertex lies on neighbor's faces
          if (hanging_vertices.find(v) == hanging_vertices.end()) {
            auto faces = get_cell_faces(mesh, neighbor_id);
            for (const auto& face : faces) {
              if (vertex_on_face(mesh, v, face)) {
                HangingVertexInfo info;
                info.vertex_id = v;
                info.parent_type = ConstraintParentType::Face;
                info.parent_vertices = face;
                compute_barycentric_weights(mesh, v, face, info.weights);
                info.coarse_neighbors.push_back(neighbor_id);
                hanging_vertices[v] = info;
                break;
              }
            }
          }
        }
      }
    }
  }

  return hanging_vertices;
}

std::vector<HangingVertexConstraint>
HangingVertexUtils::generate_constraints(const MeshBase& mesh,
                                        const std::unordered_map<index_t, HangingVertexInfo>& hanging_vertices) {
  std::vector<HangingVertexConstraint> constraints;

  for (const auto& [vertex_id, info] : hanging_vertices) {
    HangingVertexConstraint constraint;
    constraint.constrained_vertex = vertex_id;
    constraint.parent_type = info.parent_type;
    constraint.parent_vertices = info.parent_vertices;
    constraint.weights = info.weights;
    constraint.refinement_level = info.level_difference + 1;

    // Add adjacent cells
    auto v_cells = mesh.vertex_cells(vertex_id);
    constraint.adjacent_cells.insert(v_cells.begin(), v_cells.end());

    constraints.push_back(constraint);
  }

  return constraints;
}

std::unordered_set<index_t>
HangingVertexUtils::enforce_2to1_balance(const MeshBase& mesh,
                                        const std::unordered_set<index_t>& marked_cells,
                                        const std::vector<size_t>& refinement_levels,
                                        size_t max_iterations) {
  std::unordered_set<index_t> cells_to_refine = marked_cells;
  std::unordered_set<index_t> additional_cells;

  for (size_t iter = 0; iter < max_iterations; ++iter) {
    std::unordered_set<index_t> new_cells;

    for (index_t cell_id : cells_to_refine) {
      // Get neighbors
      auto cell_verts = mesh.cell_vertices(cell_id);
      std::unordered_set<index_t> neighbors;

      for (index_t v : cell_verts) {
        auto v_cells = mesh.vertex_cells(v);
        for (index_t n : v_cells) {
          if (n != cell_id) {
            neighbors.insert(n);
          }
        }
      }

      // Check level differences
      size_t cell_level = refinement_levels[cell_id];
      for (index_t neighbor_id : neighbors) {
        size_t neighbor_level = refinement_levels[neighbor_id];
        if (cell_level > neighbor_level + 1) {
          // Neighbor needs refinement for 2:1 balance
          if (cells_to_refine.find(neighbor_id) == cells_to_refine.end()) {
            new_cells.insert(neighbor_id);
            additional_cells.insert(neighbor_id);
          }
        }
      }
    }

    if (new_cells.empty()) break;
    cells_to_refine.insert(new_cells.begin(), new_cells.end());
  }

  return additional_cells;
}

bool HangingVertexUtils::is_hanging_vertex(const MeshBase& mesh,
                                          index_t vertex_id,
                                          const std::unordered_set<index_t>& refined_cells) {
  auto v_cells = mesh.vertex_cells(vertex_id);

  bool has_fine = false;
  bool has_coarse = false;

  for (index_t cell_id : v_cells) {
    if (refined_cells.find(cell_id) != refined_cells.end()) {
      has_fine = true;
    } else {
      has_coarse = true;
    }
  }

  return has_fine && has_coarse;
}

HangingVertexStats
HangingVertexUtils::compute_statistics(const std::unordered_map<index_t, HangingVertexInfo>& hanging_vertices) {
  HangingVertexStats stats;
  stats.num_hanging = hanging_vertices.size();

  for (const auto& [id, info] : hanging_vertices) {
    if (info.parent_type == ConstraintParentType::Edge) {
      stats.num_edge_hanging++;
    } else if (info.parent_type == ConstraintParentType::Face) {
      stats.num_face_hanging++;
    }
    stats.max_level_difference = std::max(stats.max_level_difference, info.level_difference);
  }

  return stats;
}

bool HangingVertexUtils::validate_balance(const MeshBase& mesh,
                                         const std::vector<size_t>& refinement_levels,
                                         std::vector<index_t>* violations) {
  bool is_balanced = true;

  const size_t num_cells = mesh.n_cells();
  for (index_t cell_id = 0; cell_id < static_cast<index_t>(num_cells); ++cell_id) {
    // Get neighbors
    auto cell_verts = mesh.cell_vertices(cell_id);
    std::unordered_set<index_t> neighbors;

    for (index_t v : cell_verts) {
      auto v_cells = mesh.vertex_cells(v);
      for (index_t n : v_cells) {
        if (n != cell_id) {
          neighbors.insert(n);
        }
      }
    }

    // Check level differences
    size_t cell_level = refinement_levels[cell_id];
    for (index_t neighbor_id : neighbors) {
      size_t neighbor_level = refinement_levels[neighbor_id];
      size_t diff = (cell_level > neighbor_level) ?
                    (cell_level - neighbor_level) :
                    (neighbor_level - cell_level);

      if (diff > 1) {
        is_balanced = false;
        if (violations) {
          violations->push_back(cell_id);
        }
        break;
      }
    }
  }

  return is_balanced;
}

} // namespace svmp
