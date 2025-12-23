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

#include "NodeOrdering.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace svmp {

namespace {

CellTopology::HighOrderKind deduce_kind(CellFamily family, int order, size_t node_count, int& p) {
  const int p_ser = CellTopology::infer_serendipity_order(family, node_count);
  const int p_lag = CellTopology::infer_lagrange_order(family, node_count);
  if (p_ser > 0) {
    p = p_ser;
    return CellTopology::HighOrderKind::Serendipity;
  }
  if (p_lag > 0) {
    p = p_lag;
    return CellTopology::HighOrderKind::Lagrange;
  }
  // Best-effort fallback: trust declared order and assume Lagrange.
  p = std::max(1, order);
  return CellTopology::HighOrderKind::Lagrange;
}

// -------------------------
// VTK node label generation
// -------------------------

std::vector<int> line_indices_vtk(int p) {
  std::vector<int> idx;
  idx.reserve(static_cast<size_t>(p + 1));
  for (int i = 0; i <= p; ++i) idx.push_back(i);
  return idx;
}

std::vector<std::array<int, 3>> triangle_exponents_vtk(int p) {
  std::vector<std::array<int, 3>> exps;
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
      std::array<int, 3> e = {0, 0, 0};
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

std::vector<std::array<int, 2>> quad_indices_vtk(int p, CellTopology::HighOrderKind kind) {
  std::vector<std::array<int, 2>> idx;
  idx.reserve(static_cast<size_t>((p + 1) * (p + 1)));

  idx.push_back({0, 0});
  idx.push_back({p, 0});
  idx.push_back({p, p});
  idx.push_back({0, p});

  const auto eview = CellTopology::get_edges_view(CellFamily::Quad);
  const int steps = std::max(0, p - 1);
  const std::array<std::array<int, 2>, 4> corner_grid = {{{0, 0}, {p, 0}, {p, p}, {0, p}}};
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

  if (kind == CellTopology::HighOrderKind::Lagrange) {
    for (int i = 1; i <= p - 1; ++i) {
      for (int j = 1; j <= p - 1; ++j) {
        idx.push_back({i, j});
      }
    }
  }

  return idx;
}

std::vector<std::array<int, 4>> tetra_exponents_vtk(int p) {
  std::vector<std::array<int, 4>> exps;
  exps.reserve(static_cast<size_t>((p + 1) * (p + 2) * (p + 3) / 6));

  exps.push_back({p, 0, 0, 0});
  exps.push_back({0, p, 0, 0});
  exps.push_back({0, 0, p, 0});
  exps.push_back({0, 0, 0, p});

  const auto eview = CellTopology::get_edges_view(CellFamily::Tetra);
  const int steps = std::max(0, p - 1);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = static_cast<int>(eview.pairs_flat[2 * ei + 0]);
    const int b = static_cast<int>(eview.pairs_flat[2 * ei + 1]);
    for (int k = 1; k <= steps; ++k) {
      std::array<int, 4> e = {0, 0, 0, 0};
      e[static_cast<size_t>(a)] = p - k;
      e[static_cast<size_t>(b)] = k;
      exps.push_back(e);
    }
  }

  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Tetra);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    const int fv = e - b;
    if (fv != 3) continue;
    const int v0 = fview.indices[b + 0];
    const int v1 = fview.indices[b + 1];
    const int v2 = fview.indices[b + 2];
    for (int i = 1; i <= p - 2; ++i) {
      for (int j = 1; j <= p - 1 - i; ++j) {
        std::array<int, 4> f = {0, 0, 0, 0};
        f[static_cast<size_t>(v0)] = p - i - j;
        f[static_cast<size_t>(v1)] = i;
        f[static_cast<size_t>(v2)] = j;
        exps.push_back(f);
      }
    }
  }

  for (int i = 1; i <= p - 3; ++i) {
    for (int j = 1; j <= p - 2 - i; ++j) {
      for (int k = 1; k <= p - 1 - i - j; ++k) {
        exps.push_back({p - i - j - k, i, j, k});
      }
    }
  }

  return exps;
}

std::vector<std::array<int, 3>> hex_indices_vtk(int p, CellTopology::HighOrderKind kind) {
  std::vector<std::array<int, 3>> idx;
  idx.reserve(static_cast<size_t>((p + 1) * (p + 1) * (p + 1)));

  const std::array<std::array<int, 3>, 8> corner_grid = {{
      {0, 0, 0}, {p, 0, 0}, {p, p, 0}, {0, p, 0},
      {0, 0, p}, {p, 0, p}, {p, p, p}, {0, p, p},
  }};

  for (const auto& c : corner_grid) idx.push_back(c);

  const auto eview = CellTopology::get_edges_view(CellFamily::Hex);
  const int steps = std::max(0, p - 1);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = static_cast<int>(eview.pairs_flat[2 * ei + 0]);
    const int b = static_cast<int>(eview.pairs_flat[2 * ei + 1]);
    const auto A = corner_grid[static_cast<size_t>(a)];
    const auto B = corner_grid[static_cast<size_t>(b)];
    for (int k = 1; k <= steps; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(p);
      const int ii = static_cast<int>(std::lround((1.0 - t) * A[0] + t * B[0]));
      const int jj = static_cast<int>(std::lround((1.0 - t) * A[1] + t * B[1]));
      const int kk = static_cast<int>(std::lround((1.0 - t) * A[2] + t * B[2]));
      idx.push_back({ii, jj, kk});
    }
  }

  if (kind == CellTopology::HighOrderKind::Lagrange) {
    const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
    for (int fi = 0; fi < fview.face_count; ++fi) {
      const int b = fview.offsets[fi];
      const int e = fview.offsets[fi + 1];
      const int fv = e - b;
      if (fv != 4) continue;
      const int v0 = fview.indices[b + 0];
      const int v1 = fview.indices[b + 1];
      const int v2 = fview.indices[b + 2];
      const int v3 = fview.indices[b + 3];
      const auto A = corner_grid[static_cast<size_t>(v0)];
      const auto B = corner_grid[static_cast<size_t>(v1)];
      const auto C = corner_grid[static_cast<size_t>(v2)];
      const auto D = corner_grid[static_cast<size_t>(v3)];

      for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
          const double u = static_cast<double>(i) / static_cast<double>(p);
          const double v = static_cast<double>(j) / static_cast<double>(p);
          std::array<int, 3> g = {0, 0, 0};
          for (int d = 0; d < 3; ++d) {
            const double x =
                (1.0 - u) * (1.0 - v) * A[d] +
                u * (1.0 - v) * B[d] +
                u * v * C[d] +
                (1.0 - u) * v * D[d];
            g[static_cast<size_t>(d)] = static_cast<int>(std::lround(x));
          }
          idx.push_back(g);
        }
      }
    }

    for (int i = 1; i <= p - 1; ++i) {
      for (int j = 1; j <= p - 1; ++j) {
        for (int k = 1; k <= p - 1; ++k) {
          idx.push_back({i, j, k});
        }
      }
    }
  }

  return idx;
}

struct WedgeLabel {
  std::array<int, 3> tri_exp; // (a0,a1,a2) sum p
  int kz = 0;                 // 0..p
};

std::array<int, 3> wedge_tri_exp_corner(int corner_id, int p) {
  switch (corner_id) {
    case 0: return {p, 0, 0};
    case 1: return {0, p, 0};
    case 2: return {0, 0, p};
    case 3: return {p, 0, 0};
    case 4: return {0, p, 0};
    case 5: return {0, 0, p};
    default: return {0, 0, 0};
  }
}

int wedge_k_corner(int corner_id, int p) {
  switch (corner_id) {
    case 0: case 1: case 2: return 0;
    case 3: case 4: case 5: return p;
    default: return 0;
  }
}

WedgeLabel wedge_corner_label(int corner_id, int p) {
  return {wedge_tri_exp_corner(corner_id, p), wedge_k_corner(corner_id, p)};
}

WedgeLabel wedge_lerp(const WedgeLabel& a, const WedgeLabel& b, int step, int p) {
  WedgeLabel out;
  for (int d = 0; d < 3; ++d) {
    out.tri_exp[static_cast<size_t>(d)] =
        static_cast<int>(((p - step) * a.tri_exp[static_cast<size_t>(d)] + step * b.tri_exp[static_cast<size_t>(d)]) / p);
  }
  out.kz = static_cast<int>(((p - step) * a.kz + step * b.kz) / p);
  return out;
}

std::vector<WedgeLabel> wedge_labels_vtk(int p, CellTopology::HighOrderKind kind) {
  if (p < 1) {
    throw std::invalid_argument("NodeOrdering: wedge p must be >= 1");
  }

  const auto eview = CellTopology::get_edges_view(CellFamily::Wedge);
  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);

  auto tri_face_label = [p](const WedgeLabel& A, const WedgeLabel& B, const WedgeLabel& C, int i, int j) -> WedgeLabel {
    const int w1 = i;
    const int w2 = j;
    const int w0 = p - i - j;
    WedgeLabel out;
    for (int d = 0; d < 3; ++d) {
      const int v =
          w0 * A.tri_exp[static_cast<size_t>(d)] +
          w1 * B.tri_exp[static_cast<size_t>(d)] +
          w2 * C.tri_exp[static_cast<size_t>(d)];
      out.tri_exp[static_cast<size_t>(d)] = v / p;
    }
    out.kz = (w0 * A.kz + w1 * B.kz + w2 * C.kz) / p;
    return out;
  };

  auto quad_face_label = [p](const WedgeLabel& A, const WedgeLabel& B, const WedgeLabel& C, const WedgeLabel& D, int i, int j) -> WedgeLabel {
    const int64_t w00 = static_cast<int64_t>(p - i) * static_cast<int64_t>(p - j);
    const int64_t w10 = static_cast<int64_t>(i) * static_cast<int64_t>(p - j);
    const int64_t w11 = static_cast<int64_t>(i) * static_cast<int64_t>(j);
    const int64_t w01 = static_cast<int64_t>(p - i) * static_cast<int64_t>(j);
    const int64_t denom = static_cast<int64_t>(p) * static_cast<int64_t>(p);

    WedgeLabel out;
    for (int d = 0; d < 3; ++d) {
      const int64_t v =
          w00 * A.tri_exp[static_cast<size_t>(d)] +
          w10 * B.tri_exp[static_cast<size_t>(d)] +
          w11 * C.tri_exp[static_cast<size_t>(d)] +
          w01 * D.tri_exp[static_cast<size_t>(d)];
      out.tri_exp[static_cast<size_t>(d)] = static_cast<int>(v / denom);
    }
    const int64_t kz =
        w00 * A.kz + w10 * B.kz + w11 * C.kz + w01 * D.kz;
    out.kz = static_cast<int>(kz / denom);
    return out;
  };

  std::vector<WedgeLabel> labels;
  if (kind == CellTopology::HighOrderKind::Serendipity) {
    labels.reserve(static_cast<size_t>(6 + eview.edge_count * std::max(0, p - 1)));
  } else {
    const size_t n_lagr = static_cast<size_t>(p + 1) * static_cast<size_t>(p + 1) * static_cast<size_t>(p + 2) / 2u;
    labels.reserve(n_lagr);
  }

  // corners
  for (int c = 0; c < 6; ++c) labels.push_back(wedge_corner_label(c, p));

  // edges (edge view order)
  const int steps = std::max(0, p - 1);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = static_cast<int>(eview.pairs_flat[2 * ei + 0]);
    const int b = static_cast<int>(eview.pairs_flat[2 * ei + 1]);
    const auto A = wedge_corner_label(a, p);
    const auto B = wedge_corner_label(b, p);
    for (int k = 1; k <= steps; ++k) {
      labels.push_back(wedge_lerp(A, B, k, p));
    }
  }

  if (kind == CellTopology::HighOrderKind::Serendipity) {
    return labels;
  }

  // faces (oriented face order)
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    const int fv = e - b;
    if (fv == 3) {
      const int v0 = static_cast<int>(fview.indices[b + 0]);
      const int v1 = static_cast<int>(fview.indices[b + 1]);
      const int v2 = static_cast<int>(fview.indices[b + 2]);
      const auto A = wedge_corner_label(v0, p);
      const auto B = wedge_corner_label(v1, p);
      const auto C = wedge_corner_label(v2, p);
      for (int i = 1; i <= p - 2; ++i) {
        for (int j = 1; j <= p - 1 - i; ++j) {
          labels.push_back(tri_face_label(A, B, C, i, j));
        }
      }
    } else if (fv == 4) {
      const int v0 = static_cast<int>(fview.indices[b + 0]);
      const int v1 = static_cast<int>(fview.indices[b + 1]);
      const int v2 = static_cast<int>(fview.indices[b + 2]);
      const int v3 = static_cast<int>(fview.indices[b + 3]);
      const auto A = wedge_corner_label(v0, p);
      const auto B = wedge_corner_label(v1, p);
      const auto C = wedge_corner_label(v2, p);
      const auto D = wedge_corner_label(v3, p);
      for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
          labels.push_back(quad_face_label(A, B, C, D, i, j));
        }
      }
    }
  }

  // volume nodes
  for (int i = 1; i <= p - 2; ++i) {
    for (int j = 1; j <= p - 1 - i; ++j) {
      for (int k = 1; k <= p - 1; ++k) {
        labels.push_back({{p - i - j, i, j}, k});
      }
    }
  }

  return labels;
}

std::vector<std::array<int, 3>> pyramid_indices_vtk(int p, CellTopology::HighOrderKind kind) {
  if (p < 1) {
    throw std::invalid_argument("NodeOrdering: pyramid p must be >= 1");
  }

  std::vector<std::array<int, 3>> idx;
  if (kind == CellTopology::HighOrderKind::Serendipity) {
    idx.reserve(static_cast<size_t>(5 + 8 * std::max(0, p - 1)));
  } else {
    const size_t n_lagr =
        static_cast<size_t>(p + 1) * static_cast<size_t>(p + 2) * static_cast<size_t>(2 * p + 3) / 6u;
    idx.reserve(n_lagr);
  }

  // corners 0..3 at k=0, apex at k=p
  idx.push_back({0, 0, 0});
  idx.push_back({p, 0, 0});
  idx.push_back({p, p, 0});
  idx.push_back({0, p, 0});
  idx.push_back({0, 0, p});

  // edges (edge view order)
  const auto eview = CellTopology::get_edges_view(CellFamily::Pyramid);
  const int steps = std::max(0, p - 1);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = static_cast<int>(eview.pairs_flat[2 * ei + 0]);
    const int b = static_cast<int>(eview.pairs_flat[2 * ei + 1]);
    const auto A = idx[static_cast<size_t>(a)];
    const auto B = idx[static_cast<size_t>(b)];
    for (int k = 1; k <= steps; ++k) {
      idx.push_back({
          ((p - k) * A[0] + k * B[0]) / p,
          ((p - k) * A[1] + k * B[1]) / p,
          ((p - k) * A[2] + k * B[2]) / p,
      });
    }
  }

  if (kind == CellTopology::HighOrderKind::Serendipity) {
    return idx;
  }

  // faces (quad base + 4 tri sides), oriented face order
  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    const int fv = e - b;
    if (fv == 4) {
      const int v0 = static_cast<int>(fview.indices[b + 0]);
      const int v1 = static_cast<int>(fview.indices[b + 1]);
      const int v2 = static_cast<int>(fview.indices[b + 2]);
      const int v3 = static_cast<int>(fview.indices[b + 3]);
      const auto A = idx[static_cast<size_t>(v0)];
      const auto B = idx[static_cast<size_t>(v1)];
      const auto C = idx[static_cast<size_t>(v2)];
      const auto D = idx[static_cast<size_t>(v3)];

      for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
          const int64_t w00 = static_cast<int64_t>(p - i) * static_cast<int64_t>(p - j);
          const int64_t w10 = static_cast<int64_t>(i) * static_cast<int64_t>(p - j);
          const int64_t w11 = static_cast<int64_t>(i) * static_cast<int64_t>(j);
          const int64_t w01 = static_cast<int64_t>(p - i) * static_cast<int64_t>(j);
          const int64_t denom = static_cast<int64_t>(p) * static_cast<int64_t>(p);

          idx.push_back({
              static_cast<int>((w00 * A[0] + w10 * B[0] + w11 * C[0] + w01 * D[0]) / denom),
              static_cast<int>((w00 * A[1] + w10 * B[1] + w11 * C[1] + w01 * D[1]) / denom),
              static_cast<int>((w00 * A[2] + w10 * B[2] + w11 * C[2] + w01 * D[2]) / denom),
          });
        }
      }
    } else if (fv == 3) {
      const int v0 = static_cast<int>(fview.indices[b + 0]);
      const int v1 = static_cast<int>(fview.indices[b + 1]);
      const int v2 = static_cast<int>(fview.indices[b + 2]);
      const auto A = idx[static_cast<size_t>(v0)];
      const auto B = idx[static_cast<size_t>(v1)];
      const auto C = idx[static_cast<size_t>(v2)];

      for (int i = 1; i <= p - 2; ++i) {
        for (int j = 1; j <= p - 1 - i; ++j) {
          const int w1 = i;
          const int w2 = j;
          const int w0 = p - i - j;
          idx.push_back({
              (w0 * A[0] + w1 * B[0] + w2 * C[0]) / p,
              (w0 * A[1] + w1 * B[1] + w2 * C[1]) / p,
              (w0 * A[2] + w1 * B[2] + w2 * C[2]) / p,
          });
        }
      }
    }
  }

  // volume interior (layered)
  for (int k = 1; k <= p - 2; ++k) {
    const int m = p - k;
    for (int i = 1; i <= m - 1; ++i) {
      for (int j = 1; j <= m - 1; ++j) {
        idx.push_back({i, j, k});
      }
    }
  }

  return idx;
}

// -------------------------
// Gmsh index conversions
// -------------------------

// Barycentric (triangle) to VTK triangle index (0-based), for ref>=1.
// Algorithm adapted from the standard VTK Lagrange triangle ordering.
int barycentric_to_vtk_triangle(const std::array<int, 3>& b_in, int ref) {
  int max = ref;
  int min = 0;
  std::array<int, 3> b = b_in;
  int bmin = std::min({b[0], b[1], b[2]});
  int idx = 0;

  while (bmin > min) {
    idx += 3 * ref;
    max -= 2;
    ++min;
    ref -= 3;
    bmin = std::min({b[0], b[1], b[2]});
  }

  for (int d = 0; d < 3; ++d) {
    if (b[(d + 2) % 3] == max) {
      return idx;
    }
    ++idx;
  }

  for (int d = 0; d < 3; ++d) {
    if (b[(d + 1) % 3] == min) {
      return idx + b[d] - (min + 1);
    }
    idx += max - (min + 1);
  }

  return idx;
}

int cartesian_to_gmsh_quad(const std::array<int, 2>& ij, int ref) {
  const int i = ij[0];
  const int j = ij[1];
  const bool ibdr = (i == 0 || i == ref);
  const bool jbdr = (j == 0 || j == ref);
  if (ibdr && jbdr) {
    return (i ? (j ? 2 : 1) : (j ? 3 : 0));
  }
  int offset = 4;
  if (jbdr) {
    return offset + (j ? 3 * ref - 3 - i : i - 1);
  }
  if (ibdr) {
    return offset + (i ? ref - 1 + j - 1 : 4 * ref - 4 - j);
  }
  std::array<int, 2> sub = {i - 1, j - 1};
  offset += 4 * (ref - 1);
  return offset + cartesian_to_gmsh_quad(sub, ref - 2);
}

int cartesian_to_gmsh_hex(const std::array<int, 3>& ijk, int ref) {
  const int i = ijk[0];
  const int j = ijk[1];
  const int k = ijk[2];
  const bool ibdr = (i == 0 || i == ref);
  const bool jbdr = (j == 0 || j == ref);
  const bool kbdr = (k == 0 || k == ref);

  if (ibdr && jbdr && kbdr) {
    return (i ? (j ? (k ? 6 : 2) : (k ? 5 : 1)) : (j ? (k ? 7 : 3) : (k ? 4 : 0)));
  }

  int offset = 8;
  if (jbdr && kbdr) {
    return offset + (j ? (k ? 12 * ref - 12 - i : 6 * ref - 6 - i) : (k ? 8 * ref - 9 + i : i - 1));
  }
  if (ibdr && kbdr) {
    return offset + (k ? (i ? 10 * ref - 11 + j : 9 * ref - 10 + j) : (i ? 3 * ref - 4 + j : ref - 2 + j));
  }
  if (ibdr && jbdr) {
    return offset + (i ? (j ? 6 * ref - 7 + k : 4 * ref - 5 + k) : (j ? 7 * ref - 8 + k : 2 * ref - 3 + k));
  }

  if (ibdr) {
    const std::array<int, 2> sub = {i ? j - 1 : k - 1, i ? k - 1 : j - 1};
    offset += (12 + (i ? 3 : 2) * (ref - 1)) * (ref - 1);
    return offset + cartesian_to_gmsh_quad(sub, ref - 2);
  }
  if (jbdr) {
    const std::array<int, 2> sub = {j ? ref - i - 1 : i - 1, k - 1};
    offset += (12 + (j ? 4 : 1) * (ref - 1)) * (ref - 1);
    return offset + cartesian_to_gmsh_quad(sub, ref - 2);
  }
  if (kbdr) {
    const std::array<int, 2> sub = {k ? i - 1 : j - 1, k ? j - 1 : i - 1};
    offset += (12 + (k ? 5 : 0) * (ref - 1)) * (ref - 1);
    return offset + cartesian_to_gmsh_quad(sub, ref - 2);
  }

  const std::array<int, 3> sub = {i - 1, j - 1, k - 1};
  offset += (12 + 6 * (ref - 1)) * (ref - 1);
  return offset + cartesian_to_gmsh_hex(sub, ref - 2);
}

int barycentric_to_gmsh_tet(const std::array<int, 4>& b, int ref) {
  const int i = b[0];
  const int j = b[1];
  const int k = b[2];
  const int l = b[3];
  const bool ibdr = (i == 0);
  const bool jbdr = (j == 0);
  const bool kbdr = (k == 0);
  const bool lbdr = (l == 0);

  if (ibdr && jbdr && kbdr) {
    return 0;
  } else if (jbdr && kbdr && lbdr) {
    return 1;
  } else if (ibdr && kbdr && lbdr) {
    return 2;
  } else if (ibdr && jbdr && lbdr) {
    return 3;
  }

  int offset = 4;
  if (jbdr && kbdr) {
    return offset + i - 1;
  } else if (kbdr && lbdr) {
    return offset + ref - 1 + j - 1;
  } else if (ibdr && kbdr) {
    return offset + 2 * (ref - 1) + ref - j - 1;
  } else if (ibdr && jbdr) {
    return offset + 3 * (ref - 1) + ref - k - 1;
  } else if (ibdr && lbdr) {
    return offset + 4 * (ref - 1) + ref - k - 1;
  } else if (jbdr && lbdr) {
    return offset + 5 * (ref - 1) + ref - k - 1;
  }

  offset += 6 * (ref - 1);
  if (kbdr) {
    std::array<int, 3> bout = {j - 1, i - 1, ref - i - j - 1};
    return offset + barycentric_to_vtk_triangle(bout, ref - 3);
  } else if (jbdr) {
    std::array<int, 3> bout = {i - 1, k - 1, ref - i - k - 1};
    offset += (ref - 1) * (ref - 2) / 2;
    return offset + barycentric_to_vtk_triangle(bout, ref - 3);
  } else if (ibdr) {
    std::array<int, 3> bout = {k - 1, j - 1, ref - j - k - 1};
    offset += (ref - 1) * (ref - 2);
    return offset + barycentric_to_vtk_triangle(bout, ref - 3);
  } else if (lbdr) {
    std::array<int, 3> bout = {ref - j - k - 1, j - 1, k - 1};
    offset += 3 * (ref - 1) * (ref - 2) / 2;
    return offset + barycentric_to_vtk_triangle(bout, ref - 3);
  }

  std::array<int, 4> sub = {i - 1, j - 1, k - 1, ref - i - j - k - 1};
  offset += 2 * (ref - 1) * (ref - 2);
  return offset + barycentric_to_gmsh_tet(sub, ref - 4);
}

int wedge_to_gmsh_prism(const std::array<int, 3>& ijk, int ref) {
  const int i = ijk[0];
  const int j = ijk[1];
  const int k = ijk[2];
  const int l = ref - i - j;
  const bool ibdr = (i == 0);
  const bool jbdr = (j == 0);
  const bool kbdr = (k == 0 || k == ref);
  const bool lbdr = (l == 0);

  if (ibdr && jbdr && kbdr) {
    return k ? 3 : 0;
  } else if (jbdr && lbdr && kbdr) {
    return k ? 4 : 1;
  } else if (ibdr && lbdr && kbdr) {
    return k ? 5 : 2;
  }

  int offset = 6;
  if (jbdr && kbdr) {
    return offset + (k ? 6 * (ref - 1) + i - 1 : i - 1);
  } else if (ibdr && kbdr) {
    return offset + (k ? 7 * (ref - 1) + j - 1 : ref - 1 + j - 1);
  } else if (ibdr && jbdr) {
    return offset + 2 * (ref - 1) + k - 1;
  } else if (lbdr && kbdr) {
    return offset + (k ? 8 * (ref - 1) + j - 1 : 3 * (ref - 1) + j - 1);
  } else if (jbdr && lbdr) {
    return offset + 4 * (ref - 1) + k - 1;
  } else if (ibdr && lbdr) {
    return offset + 5 * (ref - 1) + k - 1;
  }

  offset += 9 * (ref - 1);
  if (kbdr) {
    std::array<int, 3> bout = {k ? i - 1 : j - 1, k ? j - 1 : i - 1, ref - i - j - 1};
    offset += k ? (ref - 1) * (ref - 2) / 2 : 0;
    return offset + barycentric_to_vtk_triangle(bout, ref - 3);
  }

  offset += (ref - 1) * (ref - 2);
  if (jbdr) {
    const std::array<int, 2> sub = {i - 1, k - 1};
    return offset + cartesian_to_gmsh_quad(sub, ref - 2);
  } else if (ibdr) {
    const std::array<int, 2> sub = {k - 1, j - 1};
    offset += (ref - 1) * (ref - 1);
    return offset + cartesian_to_gmsh_quad(sub, ref - 2);
  } else if (lbdr) {
    const std::array<int, 2> sub = {j - 1, k - 1};
    offset += 2 * (ref - 1) * (ref - 1);
    return offset + cartesian_to_gmsh_quad(sub, ref - 2);
  }

  offset += 3 * (ref - 1) * (ref - 1);
  std::array<int, 3> bout = {i - 1, j - 1, ref - i - j - 1};
  const int ot = barycentric_to_vtk_triangle(bout, ref - 3);
  const int os = (k == 1) ? 0 : (k == ref - 1 ? 1 : k);
  return offset + (ref - 1) * ot + os;
}

int cartesian_to_gmsh_pyramid(const std::array<int, 3>& ijk, int ref) {
  const int i = ijk[0];
  const int j = ijk[1];
  const int k = ijk[2];
  const bool ibdr = (i == 0 || i == ref - k);
  const bool jbdr = (j == 0 || j == ref - k);
  const bool kbdr = (k == 0);

  if (ibdr && jbdr && kbdr) {
    return i ? (j ? 2 : 1) : (j ? 3 : 0);
  } else if (k == ref) {
    return 4;
  }

  int offset = 5;
  if (jbdr && kbdr) {
    return offset + (j ? (6 * ref - 6 - i) : (i - 1));
  } else if (ibdr && kbdr) {
    return offset + (i ? (3 * ref - 4 + j) : (ref - 2 + j));
  } else if (ibdr && jbdr) {
    return offset + (i ? (j ? 6 : 4) : (j ? 7 : 2)) * (ref - 1) + k - 1;
  }

  offset += 8 * (ref - 1);
  if (jbdr) {
    std::array<int, 3> bout = {j ? ref - i - k - 1 : i - 1, k - 1, j ? i - 1 : ref - i - k - 1};
    offset += (j ? 3 : 0) * (ref - 1) * (ref - 2) / 2;
    return offset + barycentric_to_vtk_triangle(bout, ref - 3);
  } else if (ibdr) {
    std::array<int, 3> bout = {i ? j - 1 : ref - j - k - 1, k - 1, i ? ref - j - k - 1 : j - 1};
    offset += (i ? 2 : 1) * (ref - 1) * (ref - 2) / 2;
    return offset + barycentric_to_vtk_triangle(bout, ref - 3);
  } else if (kbdr) {
    const std::array<int, 2> sub = {k ? i - 1 : j - 1, k ? j - 1 : i - 1};
    offset += 2 * (ref - 1) * (ref - 2);
    return offset + cartesian_to_gmsh_quad(sub, ref - 2);
  }

  offset += (2 * (ref - 2) + (ref - 1)) * (ref - 1);
  const std::array<int, 3> sub = {i - 1, j - 1, k - 1};
  return offset + cartesian_to_gmsh_pyramid(sub, ref - 3);
}

template <typename KeyT>
void validate_permutation(const std::vector<KeyT>& perm, size_t n) {
  if (perm.size() != n) {
    throw std::runtime_error("NodeOrdering: permutation size mismatch");
  }
  std::vector<char> seen(n, 0);
  for (auto v : perm) {
    if (v < 0 || static_cast<size_t>(v) >= n) {
      throw std::runtime_error("NodeOrdering: permutation index out of range");
    }
    if (++seen[static_cast<size_t>(v)] != 1) {
      throw std::runtime_error("NodeOrdering: permutation is not one-to-one");
    }
  }
}

} // namespace

std::vector<index_t> NodeOrdering::permutation_to_vtk(NodeOrderingFormat fmt,
                                                      CellFamily family,
                                                      int order,
                                                      size_t node_count) {
  if (fmt == NodeOrderingFormat::VTK) {
    std::vector<index_t> perm(node_count, 0);
    for (index_t i = 0; i < static_cast<index_t>(node_count); ++i) perm[static_cast<size_t>(i)] = i;
    return perm;
  }

  if (fmt != NodeOrderingFormat::Gmsh) {
    throw std::invalid_argument("NodeOrdering::permutation_to_vtk: unsupported format");
  }

  int p = 1;
  const auto kind = deduce_kind(family, order, node_count, p);

  std::vector<index_t> perm;
  perm.reserve(node_count);

  if (family == CellFamily::Line) {
    // Gmsh: [v0, v1, interior...] ; VTK: [v0, interior..., v1]
    if (p < 1 || node_count != static_cast<size_t>(p + 1)) {
      throw std::runtime_error("NodeOrdering: unexpected line node count");
    }
    for (int i = 0; i <= p; ++i) {
      if (i == 0) perm.push_back(0);
      else if (i == p) perm.push_back(1);
      else perm.push_back(static_cast<index_t>(i + 1));
    }
    validate_permutation(perm, node_count);
    return perm;
  }

  if (family == CellFamily::Triangle) {
    // Gmsh triangle ordering matches VTK for the supported element types (Triangle6/10/...)
    // so we keep identity (validated by unit tests in IO/CurvilinearEval).
    std::vector<index_t> id(node_count, 0);
    for (index_t i = 0; i < static_cast<index_t>(node_count); ++i) id[static_cast<size_t>(i)] = i;
    return id;
  }

  if (family == CellFamily::Quad) {
    const auto idx = quad_indices_vtk(p, kind);
    if (idx.size() != node_count) {
      throw std::runtime_error("NodeOrdering: quad node count mismatch");
    }
    for (const auto& ij : idx) {
      perm.push_back(static_cast<index_t>(cartesian_to_gmsh_quad(ij, p)));
    }
    validate_permutation(perm, node_count);
    return perm;
  }

  if (family == CellFamily::Tetra) {
    const auto exps = tetra_exponents_vtk(p);
    if (exps.size() != node_count) {
      throw std::runtime_error("NodeOrdering: tetra node count mismatch");
    }
    for (const auto& e : exps) {
      perm.push_back(static_cast<index_t>(barycentric_to_gmsh_tet(e, p)));
    }
    validate_permutation(perm, node_count);
    return perm;
  }

  if (family == CellFamily::Hex) {
    const auto idx = hex_indices_vtk(p, kind);
    if (idx.size() != node_count) {
      throw std::runtime_error("NodeOrdering: hex node count mismatch");
    }
    for (const auto& ijk : idx) {
      perm.push_back(static_cast<index_t>(cartesian_to_gmsh_hex(ijk, p)));
    }
    validate_permutation(perm, node_count);
    return perm;
  }

  if (family == CellFamily::Wedge) {
    const auto labels = wedge_labels_vtk(p, kind);
    if (labels.size() != node_count) {
      throw std::runtime_error("NodeOrdering: wedge node count mismatch");
    }
    for (const auto& lab : labels) {
      const std::array<int, 3> ijk = {lab.tri_exp[1], lab.tri_exp[2], lab.kz};
      perm.push_back(static_cast<index_t>(wedge_to_gmsh_prism(ijk, p)));
    }
    validate_permutation(perm, node_count);
    return perm;
  }

  if (family == CellFamily::Pyramid) {
    const auto idx = pyramid_indices_vtk(p, kind);
    if (idx.size() != node_count) {
      throw std::runtime_error("NodeOrdering: pyramid node count mismatch");
    }
    for (const auto& ijk : idx) {
      perm.push_back(static_cast<index_t>(cartesian_to_gmsh_pyramid(ijk, p)));
    }
    validate_permutation(perm, node_count);
    return perm;
  }

  throw std::runtime_error("NodeOrdering: unsupported cell family for Gmsh mapping");
}

void NodeOrdering::reorder_to_vtk(NodeOrderingFormat fmt,
                                  CellFamily family,
                                  int order,
                                  std::vector<size_t>& nodes) {
  if (fmt == NodeOrderingFormat::VTK) return;
  const auto perm = permutation_to_vtk(fmt, family, order, nodes.size());
  std::vector<size_t> out(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    out[i] = nodes[static_cast<size_t>(perm[i])];
  }
  nodes.swap(out);
}

std::vector<index_t> NodeOrdering::permutation_from_vtk(NodeOrderingFormat fmt,
                                                        CellFamily family,
                                                        int order,
                                                        size_t node_count) {
  if (fmt == NodeOrderingFormat::VTK) {
    std::vector<index_t> perm(node_count, 0);
    for (index_t i = 0; i < static_cast<index_t>(node_count); ++i) perm[static_cast<size_t>(i)] = i;
    return perm;
  }
  const auto to_vtk = permutation_to_vtk(fmt, family, order, node_count);
  std::vector<index_t> inv(node_count, 0);
  for (index_t i = 0; i < static_cast<index_t>(node_count); ++i) {
    inv[static_cast<size_t>(to_vtk[static_cast<size_t>(i)])] = i;
  }
  validate_permutation(inv, node_count);
  return inv;
}

void NodeOrdering::reorder_from_vtk(NodeOrderingFormat fmt,
                                    CellFamily family,
                                    int order,
                                    std::vector<size_t>& nodes) {
  if (fmt == NodeOrderingFormat::VTK) return;
  const auto perm = permutation_from_vtk(fmt, family, order, nodes.size());
  std::vector<size_t> out(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    out[i] = nodes[static_cast<size_t>(perm[i])];
  }
  nodes.swap(out);
}

} // namespace svmp
