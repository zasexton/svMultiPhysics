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

/**
 * @file test_MeshOrientation.cpp
 * @brief Unit tests for MeshOrientation / OrientationManager
 */

#include "Geometry/MeshOrientation.h"
#include "Core/MeshBase.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                           \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n"; \
      std::abort();                                                                            \
    }                                                                                          \
  } while (0)

static MeshBase create_two_hex_mesh() {
  MeshBase mesh;

  // Structured 2x1x1 hex mesh (2 cells, 12 vertices)
  const int nx = 2, ny = 1, nz = 1;
  std::vector<real_t> coords;
  coords.reserve(static_cast<size_t>((nx + 1) * (ny + 1) * (nz + 1)) * 3);

  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        coords.push_back(static_cast<real_t>(i));
        coords.push_back(static_cast<real_t>(j));
        coords.push_back(static_cast<real_t>(k));
      }
    }
  }

  std::vector<index_t> conn;
  std::vector<offset_t> offsets = {0};

  // Cell 0 (i = 0)
  conn.insert(conn.end(), {0, 1, 4, 3, 6, 7, 10, 9});
  offsets.push_back(static_cast<offset_t>(conn.size()));

  // Cell 1 (i = 1)
  conn.insert(conn.end(), {1, 2, 5, 4, 7, 8, 11, 10});
  offsets.push_back(static_cast<offset_t>(conn.size()));

  std::vector<CellShape> shapes = {{CellFamily::Hex, 8, 1}, {CellFamily::Hex, 8, 1}};

  mesh.build_from_arrays(3, coords, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

static std::array<gid_t, 4> canonicalize_quad(const std::array<gid_t, 4>& local) {
  auto rotate = [](const std::array<gid_t, 4>& v, int s) {
    std::array<gid_t, 4> out{};
    for (int i = 0; i < 4; ++i) out[static_cast<size_t>(i)] = v[static_cast<size_t>((i + s) % 4)];
    return out;
  };

  auto best = rotate(local, 0);
  for (int s = 1; s < 4; ++s) {
    auto cand = rotate(local, s);
    if (std::lexicographical_compare(cand.begin(), cand.end(), best.begin(), best.end())) best = cand;
  }

  std::array<gid_t, 4> rev{{local[3], local[2], local[1], local[0]}};
  for (int s = 0; s < 4; ++s) {
    auto cand = rotate(rev, s);
    if (std::lexicographical_compare(cand.begin(), cand.end(), best.begin(), best.end())) best = cand;
  }

  return best;
}

static perm_code_t expected_quad_code(const std::array<gid_t, 4>& canonical, const std::array<gid_t, 4>& local) {
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

static perm_code_t expected_edge_code(const std::array<gid_t, 2>& canonical, const std::array<gid_t, 2>& local) {
  for (perm_code_t code = 0; code < EdgePermutation::num_orientations(); ++code) {
    auto perm = EdgePermutation::apply(code, 0, 1);
    if (canonical[static_cast<size_t>(perm[0])] == local[0] &&
        canonical[static_cast<size_t>(perm[1])] == local[1]) {
      return code;
    }
  }
  return -1;
}

static void test_hex_face_and_edge_orientations() {
  auto mesh = create_two_hex_mesh();
  OrientationManager orient(mesh);
  orient.build();

  const auto& vertex_gids = mesh.vertex_gids();

  const auto face_view = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
  ASSERT(face_view.face_count == 6);
  const auto edge_view = CellTopology::get_edges_view(CellFamily::Hex);
  ASSERT(edge_view.edge_count == 12);

  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    auto [cell_verts, n] = mesh.cell_vertices_span(c);
    (void)n;

    // Faces (quads)
    for (int lf = 0; lf < face_view.face_count; ++lf) {
      const int start = face_view.offsets[lf];
      const int end = face_view.offsets[lf + 1];
      ASSERT(end - start == 4);

      std::array<gid_t, 4> local{};
      for (int i = 0; i < 4; ++i) {
        const index_t lvi = face_view.indices[start + i];
        const index_t vi = cell_verts[lvi];
        local[static_cast<size_t>(i)] = vertex_gids[static_cast<size_t>(vi)];
      }

      const auto canonical = canonicalize_quad(local);
      const perm_code_t expected = expected_quad_code(canonical, local);
      const perm_code_t actual = orient.face_orientation(c, lf);

      ASSERT(expected >= 0);
      ASSERT(actual == expected);
    }

    // Edges
    for (int le = 0; le < edge_view.edge_count; ++le) {
      const index_t l0 = edge_view.pairs_flat[2 * le + 0];
      const index_t l1 = edge_view.pairs_flat[2 * le + 1];
      const index_t v0 = cell_verts[l0];
      const index_t v1 = cell_verts[l1];

      std::array<gid_t, 2> local = {vertex_gids[static_cast<size_t>(v0)],
                                    vertex_gids[static_cast<size_t>(v1)]};
      std::array<gid_t, 2> canonical = local;
      if (canonical[0] > canonical[1]) std::swap(canonical[0], canonical[1]);

      const perm_code_t expected = expected_edge_code(canonical, local);
      const perm_code_t actual = orient.edge_orientation(c, le);

      ASSERT(expected >= 0);
      ASSERT(actual == expected);
    }
  }
}

} // namespace svmp::test

int main() {
  svmp::test::test_hex_face_and_edge_orientations();
  std::cout << "MeshOrientation tests PASSED\n";
  return 0;
}
