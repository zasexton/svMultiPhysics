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

#ifndef SVMP_MESH_ORIENTATION_H
#define SVMP_MESH_ORIENTATION_H

#include "../Core/MeshBase.h"
#include "../Topology/CellTopology.h"
#include <array>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace svmp {

// ====================
// P0 #2: Sub-entity Orientation & Permutation Codes
// ====================
// Small integers encoding the mapping from canonical edge/face orientation
// to the element's local orientation. Essential for:
//   - Correct glueing of FE shape functions across faces (CG/DG)
//   - Mortar/Nitsche coupling with consistent normals
//   - Conservative transfers across non-matching meshes
//   - Consistent boundary condition application

using perm_code_t = int8_t; // -1 = not computed, 0+ = permutation index

// ====================
// Edge Permutations
// ====================
// For an edge with vertices [v0, v1], there are 2 possible orientations:
//   Code 0: [v0, v1] (canonical)
//   Code 1: [v1, v0] (reversed)

class EdgePermutation {
public:
  static constexpr int num_orientations() { return 2; }

  static std::array<index_t, 2> apply(perm_code_t code, index_t v0, index_t v1) {
    if (code == 0) return {v0, v1};
    if (code == 1) return {v1, v0};
    throw std::runtime_error("EdgePermutation: invalid code " + std::to_string(code));
  }

  static std::vector<index_t> apply(perm_code_t code, const std::vector<index_t>& vertices) {
    if (vertices.size() != 2) {
      throw std::runtime_error("EdgePermutation: expected 2 vertices, got " + std::to_string(vertices.size()));
    }
    auto result = apply(code, vertices[0], vertices[1]);
    return {result[0], result[1]};
  }

  // Compute the permutation code that maps from canonical to local
  static perm_code_t compute(const std::array<index_t, 2>& canonical, const std::array<index_t, 2>& local) {
    if (canonical[0] == local[0] && canonical[1] == local[1]) return 0;
    if (canonical[0] == local[1] && canonical[1] == local[0]) return 1;
    throw std::runtime_error("EdgePermutation: local vertices don't match canonical");
  }
};

// ====================
// Face Permutations (Triangle)
// ====================
// For a triangle with vertices [v0, v1, v2], there are 6 possible orientations:
//   3 rotations × 2 reflections
// Canonical ordering assumes counter-clockwise when viewed from "outside"

class TrianglePermutation {
public:
  static constexpr int num_orientations() { return 6; }

  static std::array<index_t, 3> apply(perm_code_t code, index_t v0, index_t v1, index_t v2) {
    switch (code) {
      case 0: return {v0, v1, v2}; // identity
      case 1: return {v1, v2, v0}; // rotate +120°
      case 2: return {v2, v0, v1}; // rotate +240°
      case 3: return {v0, v2, v1}; // reflect across v0-midpoint
      case 4: return {v2, v1, v0}; // reflect across v2-midpoint
      case 5: return {v1, v0, v2}; // reflect across v1-midpoint
      default:
        throw std::runtime_error("TrianglePermutation: invalid code " + std::to_string(code));
    }
  }

  static std::vector<index_t> apply(perm_code_t code, const std::vector<index_t>& vertices) {
    if (vertices.size() != 3) {
      throw std::runtime_error("TrianglePermutation: expected 3 vertices, got " + std::to_string(vertices.size()));
    }
    auto result = apply(code, vertices[0], vertices[1], vertices[2]);
    return {result[0], result[1], result[2]};
  }

  // Compute the permutation code (simplified version; full version requires checking all 6)
  static perm_code_t compute(const std::array<index_t, 3>& canonical, const std::array<index_t, 3>& local) {
    // Try all 6 permutations
    for (perm_code_t code = 0; code < 6; ++code) {
      auto perm = apply(code, canonical[0], canonical[1], canonical[2]);
      if (perm[0] == local[0] && perm[1] == local[1] && perm[2] == local[2]) {
        return code;
      }
    }
    throw std::runtime_error("TrianglePermutation: local vertices don't match any permutation of canonical");
  }
};

// ====================
// Face Permutations (Quad)
// ====================
// For a quad with vertices [v0, v1, v2, v3], there are 8 possible orientations:
//   4 rotations × 2 reflections

class QuadPermutation {
public:
  static constexpr int num_orientations() { return 8; }

  static std::array<index_t, 4> apply(perm_code_t code, index_t v0, index_t v1, index_t v2, index_t v3) {
    switch (code) {
      case 0: return {v0, v1, v2, v3}; // identity
      case 1: return {v1, v2, v3, v0}; // rotate +90°
      case 2: return {v2, v3, v0, v1}; // rotate +180°
      case 3: return {v3, v0, v1, v2}; // rotate +270°
      case 4: return {v0, v3, v2, v1}; // reflect across v0-v2 diagonal
      case 5: return {v1, v0, v3, v2}; // reflect across v1-v3 diagonal
      case 6: return {v3, v2, v1, v0}; // reflect + rotate
      case 7: return {v2, v1, v0, v3}; // reflect + rotate
      default:
        throw std::runtime_error("QuadPermutation: invalid code " + std::to_string(code));
    }
  }

  static std::vector<index_t> apply(perm_code_t code, const std::vector<index_t>& vertices) {
    if (vertices.size() != 4) {
      throw std::runtime_error("QuadPermutation: expected 4 vertices, got " + std::to_string(vertices.size()));
    }
    auto result = apply(code, vertices[0], vertices[1], vertices[2], vertices[3]);
    return {result[0], result[1], result[2], result[3]};
  }

  static perm_code_t compute(const std::array<index_t, 4>& canonical, const std::array<index_t, 4>& local) {
    for (perm_code_t code = 0; code < 8; ++code) {
      auto perm = apply(code, canonical[0], canonical[1], canonical[2], canonical[3]);
      if (perm[0] == local[0] && perm[1] == local[1] && perm[2] == local[2] && perm[3] == local[3]) {
        return code;
      }
    }
    throw std::runtime_error("QuadPermutation: local vertices don't match any permutation of canonical");
  }
};

// ====================
// Orientation Manager
// ====================
// Stores and manages orientation codes for all edges/faces in a mesh.
// Builds orientation data from mesh topology.

class OrientationManager {
public:
  explicit OrientationManager(const MeshBase& mesh) : mesh_(mesh) {}

  // Build orientation data for all cells
  void build() {
    build_face_orientations();
    build_edge_orientations();
  }

  // Query face orientation for a given cell and local face index
  perm_code_t face_orientation(index_t cell_id, int local_face_id) const {
    if (cell_id < 0 || local_face_id < 0) return -1;
    const size_t c = static_cast<size_t>(cell_id);
    if (c + 1 >= face_offsets_.size()) return -1;
    const size_t start = face_offsets_[c];
    const size_t end = face_offsets_[c + 1];
    const size_t lf = static_cast<size_t>(local_face_id);
    if (start + lf >= end) return -1;
    return face_perm_[start + lf];
  }

  // Query edge orientation for a given cell and local edge index
  perm_code_t edge_orientation(index_t cell_id, int local_edge_id) const {
    if (cell_id < 0 || local_edge_id < 0) return -1;
    const size_t c = static_cast<size_t>(cell_id);
    if (c + 1 >= edge_offsets_.size()) return -1;
    const size_t start = edge_offsets_[c];
    const size_t end = edge_offsets_[c + 1];
    const size_t le = static_cast<size_t>(local_edge_id);
    if (start + le >= end) return -1;
    return edge_perm_[start + le];
  }

  // Apply face permutation to DOF indices
  template <typename T>
  std::vector<T> apply_face_permutation(index_t cell_id, int local_face_id, const std::vector<T>& canonical_dofs) const {
    perm_code_t code = face_orientation(cell_id, local_face_id);
    if (code < 0) return canonical_dofs; // no orientation data, return as-is

    // Determine face type from mesh
    const auto& shape = mesh_.cell_shape(cell_id);

    // Apply appropriate permutation based on face type
    // (simplified; production code would query actual face shape)
    if (canonical_dofs.size() == 3) {
      // Triangle face
      auto perm = TrianglePermutation::apply(code, 0, 1, 2);
      std::vector<T> result(3);
      for (int i = 0; i < 3; ++i) result[i] = canonical_dofs[perm[i]];
      return result;
    } else if (canonical_dofs.size() == 4) {
      // Quad face
      auto perm = QuadPermutation::apply(code, 0, 1, 2, 3);
      std::vector<T> result(4);
      for (int i = 0; i < 4; ++i) result[i] = canonical_dofs[perm[i]];
      return result;
    }

    return canonical_dofs;
  }

private:
  const MeshBase& mesh_;
  std::vector<size_t> face_offsets_; // per cell: [start,end)
  std::vector<size_t> edge_offsets_; // per cell: [start,end)
  std::vector<perm_code_t> face_perm_;  // per cell, per face (flattened)
  std::vector<perm_code_t> edge_perm_;  // per cell, per edge (flattened)

  template <size_t N>
  static bool lexicographically_less(const std::array<gid_t, N>& a, const std::array<gid_t, N>& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
  }

  template <size_t N>
  static std::array<gid_t, N> rotate_cycle(const std::array<gid_t, N>& v, size_t shift) {
    std::array<gid_t, N> out{};
    for (size_t i = 0; i < N; ++i) {
      out[i] = v[(i + shift) % N];
    }
    return out;
  }

  template <size_t N>
  static std::array<gid_t, N> canonicalize_cycle(const std::array<gid_t, N>& v) {
    std::array<gid_t, N> best = rotate_cycle(v, 0);
    for (size_t s = 1; s < N; ++s) {
      auto cand = rotate_cycle(v, s);
      if (lexicographically_less(cand, best)) best = cand;
    }

    std::array<gid_t, N> rev{};
    for (size_t i = 0; i < N; ++i) rev[i] = v[N - 1 - i];
    for (size_t s = 0; s < N; ++s) {
      auto cand = rotate_cycle(rev, s);
      if (lexicographically_less(cand, best)) best = cand;
    }

    return best;
  }

  static perm_code_t compute_edge_code(const std::array<gid_t,2>& canonical,
                                       const std::array<gid_t,2>& local) {
    for (perm_code_t code = 0; code < EdgePermutation::num_orientations(); ++code) {
      auto perm = EdgePermutation::apply(code, 0, 1);
      if (canonical[static_cast<size_t>(perm[0])] == local[0] &&
          canonical[static_cast<size_t>(perm[1])] == local[1]) {
        return code;
      }
    }
    return -1;
  }

  static perm_code_t compute_tri_code(const std::array<gid_t,3>& canonical,
                                      const std::array<gid_t,3>& local) {
    for (perm_code_t code = 0; code < TrianglePermutation::num_orientations(); ++code) {
      auto perm = TrianglePermutation::apply(code, 0, 1, 2);
      if (canonical[static_cast<size_t>(perm[0])] == local[0] &&
          canonical[static_cast<size_t>(perm[1])] == local[1] &&
          canonical[static_cast<size_t>(perm[2])] == local[2]) {
        return code;
      }
    }
    return -1;
  }

  static perm_code_t compute_quad_code(const std::array<gid_t,4>& canonical,
                                       const std::array<gid_t,4>& local) {
    for (perm_code_t code = 0; code < QuadPermutation::num_orientations(); ++code) {
      auto perm = QuadPermutation::apply(code, 0, 1, 2, 3);
      if (canonical[static_cast<size_t>(perm[0])] == local[0] &&
          canonical[static_cast<size_t>(perm[1])] == local[1] &&
          canonical[static_cast<size_t>(perm[2])] == local[2] &&
          canonical[static_cast<size_t>(perm[3])] == local[3]) {
        return code;
      }
    }
    return -1;
  }

  void build_face_orientations() {
    const size_t n_cells = mesh_.n_cells();
    face_offsets_.assign(n_cells + 1, 0);

    // Prefix sum to size storage.
    for (size_t c = 0; c < n_cells; ++c) {
      const auto& shape = mesh_.cell_shape(static_cast<index_t>(c));
      size_t n_faces = 0;
      try {
        auto view = CellTopology::get_oriented_boundary_faces_view(shape.family);
        n_faces = static_cast<size_t>(view.face_count);
      } catch (...) {
        n_faces = 0;
      }
      face_offsets_[c + 1] = face_offsets_[c] + n_faces;
    }

    face_perm_.assign(face_offsets_.back(), -1);

    const auto& vertex_gids = mesh_.vertex_gids();

    // Fill orientation codes per cell face.
    for (size_t c = 0; c < n_cells; ++c) {
      const auto& shape = mesh_.cell_shape(static_cast<index_t>(c));
      CellTopology::FaceListView view{};
      try {
        view = CellTopology::get_oriented_boundary_faces_view(shape.family);
      } catch (...) {
        continue;
      }

      auto [cell_vertices, n_cell_vertices] = mesh_.cell_vertices_span(static_cast<index_t>(c));
      (void)n_cell_vertices;

      for (int lf = 0; lf < view.face_count; ++lf) {
        const int start = view.offsets[lf];
        const int end = view.offsets[lf + 1];
        const int nfv = end - start;

        const size_t out_idx = face_offsets_[c] + static_cast<size_t>(lf);

        if (nfv == 3) {
          std::array<gid_t,3> local{};
          for (int i = 0; i < 3; ++i) {
            const index_t lvi = view.indices[start + i];
            const index_t vi = cell_vertices[lvi];
            local[static_cast<size_t>(i)] = vertex_gids[static_cast<size_t>(vi)];
          }
          const auto canonical = canonicalize_cycle(local);
          face_perm_[out_idx] = compute_tri_code(canonical, local);
        } else if (nfv == 4) {
          std::array<gid_t,4> local{};
          for (int i = 0; i < 4; ++i) {
            const index_t lvi = view.indices[start + i];
            const index_t vi = cell_vertices[lvi];
            local[static_cast<size_t>(i)] = vertex_gids[static_cast<size_t>(vi)];
          }
          const auto canonical = canonicalize_cycle(local);
          face_perm_[out_idx] = compute_quad_code(canonical, local);
        } else {
          // Unsupported face arity (polygonal/polyhedral) for now.
          face_perm_[out_idx] = -1;
        }
      }
    }
  }

  void build_edge_orientations() {
    const size_t n_cells = mesh_.n_cells();
    edge_offsets_.assign(n_cells + 1, 0);

    // Prefix sum to size storage.
    for (size_t c = 0; c < n_cells; ++c) {
      const auto& shape = mesh_.cell_shape(static_cast<index_t>(c));
      size_t n_edges = 0;
      try {
        auto view = CellTopology::get_edges_view(shape.family);
        n_edges = static_cast<size_t>(view.edge_count);
      } catch (...) {
        n_edges = 0;
      }
      edge_offsets_[c + 1] = edge_offsets_[c] + n_edges;
    }

    edge_perm_.assign(edge_offsets_.back(), -1);

    const auto& vertex_gids = mesh_.vertex_gids();

    for (size_t c = 0; c < n_cells; ++c) {
      const auto& shape = mesh_.cell_shape(static_cast<index_t>(c));
      CellTopology::EdgeListView view{};
      try {
        view = CellTopology::get_edges_view(shape.family);
      } catch (...) {
        continue;
      }

      auto [cell_vertices, n_cell_vertices] = mesh_.cell_vertices_span(static_cast<index_t>(c));
      (void)n_cell_vertices;

      for (int le = 0; le < view.edge_count; ++le) {
        const index_t l0 = view.pairs_flat[2 * le + 0];
        const index_t l1 = view.pairs_flat[2 * le + 1];
        const index_t v0 = cell_vertices[l0];
        const index_t v1 = cell_vertices[l1];

        std::array<gid_t,2> local = {
            vertex_gids[static_cast<size_t>(v0)],
            vertex_gids[static_cast<size_t>(v1)]
        };
        std::array<gid_t,2> canonical = local;
        if (canonical[0] > canonical[1]) std::swap(canonical[0], canonical[1]);

        const size_t out_idx = edge_offsets_[c] + static_cast<size_t>(le);
        edge_perm_[out_idx] = compute_edge_code(canonical, local);
      }
    }
  }
};

} // namespace svmp

#endif // SVMP_MESH_ORIENTATION_H
