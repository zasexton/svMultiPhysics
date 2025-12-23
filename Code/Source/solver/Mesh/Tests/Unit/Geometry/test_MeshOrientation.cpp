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

static std::vector<std::array<int,2>> quad_lagrange_indices_vtk(int p, bool include_interior) {
  std::vector<std::array<int,2>> idx;
  idx.reserve(static_cast<size_t>((p + 1) * (p + 1)));

  idx.push_back({0, 0});
  idx.push_back({p, 0});
  idx.push_back({p, p});
  idx.push_back({0, p});

  const auto eview = CellTopology::get_edges_view(CellFamily::Quad);
  const int steps = std::max(0, p - 1);
  const std::array<std::array<int,2>,4> corner_grid = {{{0,0},{p,0},{p,p},{0,p}}};
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = static_cast<int>(eview.pairs_flat[2 * ei + 0]);
    const int b = static_cast<int>(eview.pairs_flat[2 * ei + 1]);
    const auto A = corner_grid[static_cast<size_t>(a)];
    const auto B = corner_grid[static_cast<size_t>(b)];
    for (int k = 1; k <= steps; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(p);
      const int ii = static_cast<int>(std::lround((1.0 - t) * A[0] + t * B[0]));
      const int jj = static_cast<int>(std::lround((1.0 - t) * A[1] + t * B[1]));
      idx.push_back({ii, jj});
    }
  }

  if (include_interior) {
    for (int i = 1; i <= p - 1; ++i) {
      for (int j = 1; j <= p - 1; ++j) {
        idx.push_back({i, j});
      }
    }
  }
  return idx;
}

static std::vector<std::array<int,3>> triangle_exponents_vtk(int p) {
  std::vector<std::array<int,3>> exps;
  exps.reserve(static_cast<size_t>((p + 1) * (p + 2) / 2));
  exps.push_back({p, 0, 0});
  exps.push_back({0, p, 0});
  exps.push_back({0, 0, p});

  const auto eview = CellTopology::get_edges_view(CellFamily::Triangle);
  const int steps = std::max(0, p - 1);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = static_cast<int>(eview.pairs_flat[2 * ei + 0]);
    const int b = static_cast<int>(eview.pairs_flat[2 * ei + 1]);
    for (int k = 1; k <= steps; ++k) {
      std::array<int,3> e = {0, 0, 0};
      e[static_cast<size_t>(a)] = p - k;
      e[static_cast<size_t>(b)] = k;
      exps.push_back(e);
    }
  }

  for (int i = 1; i <= p - 2; ++i) {
    for (int j = 1; j <= p - 1 - i; ++j) {
      exps.push_back({p - i - j, i, j});
    }
  }
  return exps;
}

static void test_high_order_face_permutations() {
  const int p = 3;

  // ---- Quad p=3 Lagrange ----
  {
    const auto idx = quad_lagrange_indices_vtk(p, /*include_interior=*/true);
    std::vector<int> in(idx.size(), 0);
    for (size_t n = 0; n < idx.size(); ++n) {
      in[n] = 100 * idx[n][0] + idx[n][1];
    }

    auto find_idx = [&](int i, int j) -> index_t {
      for (index_t n = 0; n < static_cast<index_t>(idx.size()); ++n) {
        if (idx[static_cast<size_t>(n)][0] == i && idx[static_cast<size_t>(n)][1] == j) return n;
      }
      return INVALID_INDEX;
    };

    auto map_coord = [&](perm_code_t code, int i, int j) -> std::array<int,2> {
      switch (code) {
        case 0: return {i, j};
        case 1: return {p - j, i};
        case 2: return {p - i, p - j};
        case 3: return {j, p - i};
        case 4: return {j, i};
        case 5: return {p - i, j};
        case 6: return {i, p - j};
        case 7: return {p - j, p - i};
        default: return {0, 0};
      }
    };

    for (perm_code_t code = 0; code < QuadPermutation::num_orientations(); ++code) {
      const auto perm = HighOrderFacePermutation::quad(p, CellTopology::HighOrderKind::Lagrange, code);
      ASSERT(perm.size() == idx.size());

      std::vector<char> seen(idx.size(), 0);
      for (auto pi : perm) {
        ASSERT(pi >= 0);
        ASSERT(static_cast<size_t>(pi) < idx.size());
        seen[static_cast<size_t>(pi)]++;
      }
      for (auto s : seen) ASSERT(s == 1);

      std::vector<int> out(idx.size(), 0);
      for (size_t n = 0; n < idx.size(); ++n) out[n] = in[static_cast<size_t>(perm[n])];

      // Corners: permutation should match corner dihedral code.
      const auto corner_perm = QuadPermutation::apply(code, 0, 1, 2, 3);
      for (int k = 0; k < 4; ++k) {
        ASSERT(perm[static_cast<size_t>(k)] == corner_perm[static_cast<size_t>(k)]);
      }

      // All nodes: verify expected coordinate mapping on the (i,j) grid.
      for (size_t n = 0; n < idx.size(); ++n) {
        const int i_new = idx[n][0];
        const int j_new = idx[n][1];
        const auto [i_old, j_old] = map_coord(code, i_new, j_new);
        const index_t expected_old_idx = find_idx(i_old, j_old);
        ASSERT(expected_old_idx != INVALID_INDEX);
        ASSERT(out[n] == in[static_cast<size_t>(expected_old_idx)]);
      }
    }
  }

  // ---- Triangle p=3 (Lagrange) ----
  {
    const auto exps = triangle_exponents_vtk(p);
    std::vector<int> in(exps.size(), 0);
    for (size_t n = 0; n < exps.size(); ++n) {
      in[n] = 100 * exps[n][0] + 10 * exps[n][1] + exps[n][2];
    }

    auto find_idx = [&](const std::array<int,3>& e)->index_t {
      for (index_t n = 0; n < static_cast<index_t>(exps.size()); ++n) {
        if (exps[static_cast<size_t>(n)] == e) return n;
      }
      return INVALID_INDEX;
    };

    for (perm_code_t code = 0; code < TrianglePermutation::num_orientations(); ++code) {
      const auto perm = HighOrderFacePermutation::triangle(p, code);
      ASSERT(perm.size() == exps.size());

      std::vector<char> seen(exps.size(), 0);
      for (auto pi : perm) {
        ASSERT(pi >= 0);
        ASSERT(static_cast<size_t>(pi) < exps.size());
        seen[static_cast<size_t>(pi)]++;
      }
      for (auto s : seen) ASSERT(s == 1);

      std::vector<int> out(exps.size(), 0);
      for (size_t n = 0; n < exps.size(); ++n) out[n] = in[static_cast<size_t>(perm[n])];

      const auto corner_perm = TrianglePermutation::apply(code, 0, 1, 2);
      for (int k = 0; k < 3; ++k) {
        ASSERT(perm[static_cast<size_t>(k)] == corner_perm[static_cast<size_t>(k)]);
      }

      std::array<int,3> inv{};
      for (int k = 0; k < 3; ++k) inv[static_cast<size_t>(corner_perm[static_cast<size_t>(k)])] = k;

      for (size_t n = 0; n < exps.size(); ++n) {
        const auto& ef = exps[n];
        const std::array<int,3> ec = {
            ef[static_cast<size_t>(inv[0])],
            ef[static_cast<size_t>(inv[1])],
            ef[static_cast<size_t>(inv[2])],
        };
        const index_t expected_old = find_idx(ec);
        ASSERT(expected_old != INVALID_INDEX);
        ASSERT(out[n] == in[static_cast<size_t>(expected_old)]);
      }
    }
  }
}

} // namespace svmp::test

int main() {
  svmp::test::test_hex_face_and_edge_orientations();
  svmp::test::test_high_order_face_permutations();
  std::cout << "MeshOrientation tests PASSED\n";
  return 0;
}
