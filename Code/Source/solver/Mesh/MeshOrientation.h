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

#include "Mesh.h"
#include <array>
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
// For an edge with nodes [n0, n1], there are 2 possible orientations:
//   Code 0: [n0, n1] (canonical)
//   Code 1: [n1, n0] (reversed)

class EdgePermutation {
public:
  static constexpr int num_orientations() { return 2; }

  static std::array<index_t, 2> apply(perm_code_t code, index_t n0, index_t n1) {
    if (code == 0) return {n0, n1};
    if (code == 1) return {n1, n0};
    throw std::runtime_error("EdgePermutation: invalid code " + std::to_string(code));
  }

  static std::vector<index_t> apply(perm_code_t code, const std::vector<index_t>& nodes) {
    if (nodes.size() != 2) {
      throw std::runtime_error("EdgePermutation: expected 2 nodes, got " + std::to_string(nodes.size()));
    }
    auto result = apply(code, nodes[0], nodes[1]);
    return {result[0], result[1]};
  }

  // Compute the permutation code that maps from canonical to local
  static perm_code_t compute(const std::array<index_t, 2>& canonical, const std::array<index_t, 2>& local) {
    if (canonical[0] == local[0] && canonical[1] == local[1]) return 0;
    if (canonical[0] == local[1] && canonical[1] == local[0]) return 1;
    throw std::runtime_error("EdgePermutation: local nodes don't match canonical");
  }
};

// ====================
// Face Permutations (Triangle)
// ====================
// For a triangle with nodes [n0, n1, n2], there are 6 possible orientations:
//   3 rotations × 2 reflections
// Canonical ordering assumes counter-clockwise when viewed from "outside"

class TrianglePermutation {
public:
  static constexpr int num_orientations() { return 6; }

  static std::array<index_t, 3> apply(perm_code_t code, index_t n0, index_t n1, index_t n2) {
    switch (code) {
      case 0: return {n0, n1, n2}; // identity
      case 1: return {n1, n2, n0}; // rotate +120°
      case 2: return {n2, n0, n1}; // rotate +240°
      case 3: return {n0, n2, n1}; // reflect across n0-midpoint
      case 4: return {n2, n1, n0}; // reflect across n2-midpoint
      case 5: return {n1, n0, n2}; // reflect across n1-midpoint
      default:
        throw std::runtime_error("TrianglePermutation: invalid code " + std::to_string(code));
    }
  }

  static std::vector<index_t> apply(perm_code_t code, const std::vector<index_t>& nodes) {
    if (nodes.size() != 3) {
      throw std::runtime_error("TrianglePermutation: expected 3 nodes, got " + std::to_string(nodes.size()));
    }
    auto result = apply(code, nodes[0], nodes[1], nodes[2]);
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
    throw std::runtime_error("TrianglePermutation: local nodes don't match any permutation of canonical");
  }
};

// ====================
// Face Permutations (Quad)
// ====================
// For a quad with nodes [n0, n1, n2, n3], there are 8 possible orientations:
//   4 rotations × 2 reflections

class QuadPermutation {
public:
  static constexpr int num_orientations() { return 8; }

  static std::array<index_t, 4> apply(perm_code_t code, index_t n0, index_t n1, index_t n2, index_t n3) {
    switch (code) {
      case 0: return {n0, n1, n2, n3}; // identity
      case 1: return {n1, n2, n3, n0}; // rotate +90°
      case 2: return {n2, n3, n0, n1}; // rotate +180°
      case 3: return {n3, n0, n1, n2}; // rotate +270°
      case 4: return {n0, n3, n2, n1}; // reflect across n0-n2 diagonal
      case 5: return {n1, n0, n3, n2}; // reflect across n1-n3 diagonal
      case 6: return {n3, n2, n1, n0}; // reflect + rotate
      case 7: return {n2, n1, n0, n3}; // reflect + rotate
      default:
        throw std::runtime_error("QuadPermutation: invalid code " + std::to_string(code));
    }
  }

  static std::vector<index_t> apply(perm_code_t code, const std::vector<index_t>& nodes) {
    if (nodes.size() != 4) {
      throw std::runtime_error("QuadPermutation: expected 4 nodes, got " + std::to_string(nodes.size()));
    }
    auto result = apply(code, nodes[0], nodes[1], nodes[2], nodes[3]);
    return {result[0], result[1], result[2], result[3]};
  }

  static perm_code_t compute(const std::array<index_t, 4>& canonical, const std::array<index_t, 4>& local) {
    for (perm_code_t code = 0; code < 8; ++code) {
      auto perm = apply(code, canonical[0], canonical[1], canonical[2], canonical[3]);
      if (perm[0] == local[0] && perm[1] == local[1] && perm[2] == local[2] && perm[3] == local[3]) {
        return code;
      }
    }
    throw std::runtime_error("QuadPermutation: local nodes don't match any permutation of canonical");
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
    size_t idx = cell_face_index(cell_id, local_face_id);
    if (idx >= face_perm_.size()) return -1;
    return face_perm_[idx];
  }

  // Query edge orientation for a given cell and local edge index
  perm_code_t edge_orientation(index_t cell_id, int local_edge_id) const {
    size_t idx = cell_edge_index(cell_id, local_edge_id);
    if (idx >= edge_perm_.size()) return -1;
    return edge_perm_[idx];
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
  std::vector<perm_code_t> face_perm_;  // per cell, per face (flattened)
  std::vector<perm_code_t> edge_perm_;  // per cell, per edge (flattened)

  // Helper to compute flat index for cell+local_face
  size_t cell_face_index(index_t cell_id, int local_face_id) const {
    // Simplified: assumes fixed number of faces per cell
    // Production version would use cell_shape to determine actual face count
    return static_cast<size_t>(cell_id) * 6 + static_cast<size_t>(local_face_id);
  }

  size_t cell_edge_index(index_t cell_id, int local_edge_id) const {
    return static_cast<size_t>(cell_id) * 12 + static_cast<size_t>(local_edge_id);
  }

  void build_face_orientations() {
    // Placeholder: In production, this would:
    // 1. Loop over all cells
    // 2. For each face, determine canonical ordering (e.g., min node ID first)
    // 3. Compare to cell's local ordering
    // 4. Compute and store permutation code

    size_t n_cells = mesh_.n_cells();
    face_perm_.resize(n_cells * 6, -1); // assume max 6 faces per cell (hex)
  }

  void build_edge_orientations() {
    // Similar to face orientations
    size_t n_cells = mesh_.n_cells();
    edge_perm_.resize(n_cells * 12, -1); // assume max 12 edges per cell (hex)
  }
};

} // namespace svmp

#endif // SVMP_MESH_ORIENTATION_H
