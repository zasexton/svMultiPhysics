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

#include "PyramidBasis.h"

#include "../Topology/CellTopology.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#ifdef MESH_HAS_EIGEN
#include <Eigen/Dense>
#endif

namespace svmp {

namespace {

struct BasisIndex {
  int i = 0; // degree in r
  int j = 0; // degree in s
  int k = 0; // degree in t
  int m = 0; // i + j
};

inline size_t pyramid_lagrange_node_count(int p) {
  if (p < 1) return 0;
  return static_cast<size_t>(p + 1) * static_cast<size_t>(p + 2) * static_cast<size_t>(2 * p + 3) / 6u;
}

inline real_t clamp_pm1(real_t x) {
  if (x < -1.0) return -1.0;
  if (x > 1.0) return 1.0;
  return x;
}

void legendre_sequence(int nmax, real_t x, std::vector<real_t>& P, std::vector<real_t>& dP) {
  if (nmax < 0) {
    P.clear();
    dP.clear();
    return;
  }
  P.assign(static_cast<size_t>(nmax + 1), real_t(0));
  dP.assign(static_cast<size_t>(nmax + 1), real_t(0));

  P[0] = 1.0;
  dP[0] = 0.0;
  if (nmax == 0) return;

  P[1] = x;
  dP[1] = 1.0;

  for (int n = 1; n < nmax; ++n) {
    const real_t a = static_cast<real_t>(2 * n + 1);
    const real_t b = static_cast<real_t>(n);
    const real_t c = static_cast<real_t>(n + 1);
    P[static_cast<size_t>(n + 1)] = (a * x * P[static_cast<size_t>(n)] - b * P[static_cast<size_t>(n - 1)]) / c;
    dP[static_cast<size_t>(n + 1)] =
        (a * (P[static_cast<size_t>(n)] + x * dP[static_cast<size_t>(n)]) - b * dP[static_cast<size_t>(n - 1)]) / c;
  }
}

void jacobi_sequence(int nmax, real_t alpha, real_t beta, real_t x, std::vector<real_t>& P) {
  if (nmax < 0) {
    P.clear();
    return;
  }
  P.assign(static_cast<size_t>(nmax + 1), real_t(0));
  P[0] = 1.0;
  if (nmax == 0) return;

  P[1] = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
  for (int n = 1; n < nmax; ++n) {
    const real_t nn = static_cast<real_t>(n);
    const real_t a1 = 2.0 * (nn + 1.0) * (nn + alpha + beta + 1.0) * (2.0 * nn + alpha + beta);
    const real_t a2 = (2.0 * nn + alpha + beta + 1.0) * (alpha * alpha - beta * beta);
    const real_t a3 = (2.0 * nn + alpha + beta) * (2.0 * nn + alpha + beta + 1.0) * (2.0 * nn + alpha + beta + 2.0);
    const real_t a4 = 2.0 * (nn + alpha) * (nn + beta) * (2.0 * nn + alpha + beta + 2.0);
    P[static_cast<size_t>(n + 1)] =
        ((a2 + a3 * x) * P[static_cast<size_t>(n)] - a4 * P[static_cast<size_t>(n - 1)]) / a1;
  }
}

std::vector<BasisIndex> build_basis_indices(int p) {
  std::vector<BasisIndex> idx;
  idx.reserve(pyramid_lagrange_node_count(p));
  for (int k = 0; k <= p; ++k) {
    for (int i = 0; i <= p - k; ++i) {
      for (int j = 0; j <= p - k; ++j) {
        idx.push_back({i, j, k, i + j});
      }
    }
  }
  if (idx.size() != pyramid_lagrange_node_count(p)) {
    throw std::runtime_error("PyramidBasis: basis index count mismatch");
  }
  return idx;
}

std::vector<ParametricPoint> pyramid_reference_nodes_vtk(int p) {
  std::vector<ParametricPoint> nodes;
  nodes.reserve(pyramid_lagrange_node_count(p));

  auto corner = [](int c) -> ParametricPoint {
    switch (c) {
      case 0: return {-1, -1, 0};
      case 1: return {1, -1, 0};
      case 2: return {1, 1, 0};
      case 3: return {-1, 1, 0};
      case 4: return {0, 0, 1};
      default: return {0, 0, 0};
    }
  };

  // corners first
  for (int c = 0; c < 5; ++c) nodes.push_back(corner(c));

  // edges
  const auto eview = CellTopology::get_edges_view(CellFamily::Pyramid);
  const int steps = std::max(0, p - 1);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = eview.pairs_flat[2 * ei + 0];
    const int b = eview.pairs_flat[2 * ei + 1];
    const auto A = corner(a);
    const auto B = corner(b);
    for (int k = 1; k <= steps; ++k) {
      const real_t t = static_cast<real_t>(k) / static_cast<real_t>(p);
      nodes.push_back({(1 - t) * A[0] + t * B[0], (1 - t) * A[1] + t * B[1], (1 - t) * A[2] + t * B[2]});
    }
  }

  // faces (quad base + 4 tri sides), oriented face order
  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    const int fv = e - b;
    if (fv == 4) {
      const int v0 = fview.indices[b + 0];
      const int v1 = fview.indices[b + 1];
      const int v2 = fview.indices[b + 2];
      const int v3 = fview.indices[b + 3];
      const auto A = corner(v0);
      const auto B = corner(v1);
      const auto C = corner(v2);
      const auto D = corner(v3);
      for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
          const real_t u = static_cast<real_t>(i) / static_cast<real_t>(p);
          const real_t v = static_cast<real_t>(j) / static_cast<real_t>(p);
          nodes.push_back({
              (1 - u) * (1 - v) * A[0] + u * (1 - v) * B[0] + u * v * C[0] + (1 - u) * v * D[0],
              (1 - u) * (1 - v) * A[1] + u * (1 - v) * B[1] + u * v * C[1] + (1 - u) * v * D[1],
              (1 - u) * (1 - v) * A[2] + u * (1 - v) * B[2] + u * v * C[2] + (1 - u) * v * D[2],
          });
        }
      }
    } else if (fv == 3) {
      const int v0 = fview.indices[b + 0];
      const int v1 = fview.indices[b + 1];
      const int v2 = fview.indices[b + 2];
      const auto A = corner(v0);
      const auto B = corner(v1);
      const auto C = corner(v2);
      for (int i = 1; i <= p - 2; ++i) {
        for (int j = 1; j <= p - 1 - i; ++j) {
          const real_t w1 = static_cast<real_t>(i) / static_cast<real_t>(p);
          const real_t w2 = static_cast<real_t>(j) / static_cast<real_t>(p);
          const real_t w0 = 1.0 - w1 - w2;
          nodes.push_back({w0 * A[0] + w1 * B[0] + w2 * C[0], w0 * A[1] + w1 * B[1] + w2 * C[1],
                           w0 * A[2] + w1 * B[2] + w2 * C[2]});
        }
      }
    }
  }

  // volume interior (layered)
  for (int k = 1; k <= p - 2; ++k) {
    const int n = (p + 1) - k;
    const int m = n - 1;
    const real_t z = static_cast<real_t>(k) / static_cast<real_t>(p);
    const real_t scale = 1.0 - z;
    for (int i = 1; i <= n - 2; ++i) {
      for (int j = 1; j <= n - 2; ++j) {
        const real_t u = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(m);
        const real_t v = -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(m);
        nodes.push_back({scale * u, scale * v, z});
      }
    }
  }

  if (nodes.size() != pyramid_lagrange_node_count(p)) {
    throw std::runtime_error("PyramidBasis: node count mismatch");
  }
  return nodes;
}

// Collapsed-coordinate orthogonal basis (Bergot–Cohen–Duruflé family):
//   phi_{i,j,k}(x,y,z) = P_i(r) P_j(s) (1-z)^{i+j} * P_k^{(2(i+j)+2,0)}(t)
// where r = x/(1-z), s = y/(1-z), t = 2z-1.
void eval_bcd_basis(int p,
                    const std::vector<BasisIndex>& basis,
                    const ParametricPoint& xi,
                    std::vector<real_t>& phi,
                    std::vector<std::array<real_t, 3>>& dphi) {
  const size_t n = basis.size();
  phi.assign(n, 0.0);
  dphi.assign(n, {0, 0, 0});

  const real_t x = xi[0];
  const real_t y = xi[1];
  const real_t z = xi[2];

  const real_t f = 1.0 - z;          // (1 - z)
  const real_t t = 2.0 * z - 1.0;    // in [-1,1]
  const real_t eps = 1e-12;

  real_t r = 0.0;
  real_t s = 0.0;
  if (std::abs(f) > eps) {
    r = clamp_pm1(x / f);
    s = clamp_pm1(y / f);
  }

  std::vector<real_t> Pr, dPr, Ps, dPs;
  legendre_sequence(p, r, Pr, dPr);
  legendre_sequence(p, s, Ps, dPs);

  const int max_m = 2 * p;
  std::vector<real_t> f_pow(static_cast<size_t>(max_m + 1), 1.0);
  for (int m = 1; m <= max_m; ++m) {
    f_pow[static_cast<size_t>(m)] = f_pow[static_cast<size_t>(m - 1)] * f;
  }

  // Precompute Jacobi sequences for each m = i + j.
  std::vector<std::vector<real_t>> J(static_cast<size_t>(max_m + 1));
  std::vector<std::vector<real_t>> Jd_shift(static_cast<size_t>(max_m + 1)); // P_{k}^{(alpha+1,1)} for k=0..p-1
  for (int m = 0; m <= max_m; ++m) {
    const real_t alpha = static_cast<real_t>(2 * m + 2);
    jacobi_sequence(p, alpha, 0.0, t, J[static_cast<size_t>(m)]);
    if (p >= 1) {
      jacobi_sequence(p - 1, alpha + 1.0, 1.0, t, Jd_shift[static_cast<size_t>(m)]);
    }
  }

  for (size_t q = 0; q < n; ++q) {
    const auto& b = basis[q];
    const int i = b.i;
    const int j = b.j;
    const int k = b.k;
    const int m = b.m;

    const real_t Pi = Pr[static_cast<size_t>(i)];
    const real_t Pj = Ps[static_cast<size_t>(j)];
    const real_t dPi = dPr[static_cast<size_t>(i)];
    const real_t dPj = dPs[static_cast<size_t>(j)];

    const real_t Q = J[static_cast<size_t>(m)][static_cast<size_t>(k)];

    real_t dQdt = 0.0;
    if (k > 0) {
      const real_t factor = 0.5 * static_cast<real_t>(k + 2 * m + 3);
      dQdt = factor * Jd_shift[static_cast<size_t>(m)][static_cast<size_t>(k - 1)];
    }

    const real_t fp = f_pow[static_cast<size_t>(m)];
    const real_t fp_m1 = (m > 0) ? f_pow[static_cast<size_t>(m - 1)] : 0.0;

    phi[q] = Pi * Pj * fp * Q;

    // Handle the apex safely. For z=1, only m=0 basis functions are nonzero and depend on z via t.
    if (std::abs(f) <= eps) {
      if (m == 0) {
        dphi[q][2] = 2.0 * Pi * Pj * dQdt;
      }
      continue;
    }

    if (m > 0) {
      dphi[q][0] = dPi * Pj * fp_m1 * Q;
      dphi[q][1] = Pi * dPj * fp_m1 * Q;
    }

    if (m > 0) {
      const real_t term =
          (r * dPi * Pj + s * Pi * dPj - static_cast<real_t>(m) * Pi * Pj);
      dphi[q][2] = fp_m1 * Q * term + 2.0 * Pi * Pj * fp * dQdt;
    } else {
      dphi[q][2] = 2.0 * Pi * Pj * fp * dQdt;
    }
  }
}

void invert_square_matrix(std::vector<real_t>& A, std::vector<real_t>& inv, int n) {
  inv.assign(static_cast<size_t>(n) * static_cast<size_t>(n), real_t(0));
  for (int i = 0; i < n; ++i) inv[static_cast<size_t>(i * n + i)] = 1.0;

  auto at = [n](std::vector<real_t>& M, int r, int c) -> real_t& { return M[static_cast<size_t>(r * n + c)]; };

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    real_t max_abs = std::abs(at(A, col, col));
    for (int row = col + 1; row < n; ++row) {
      const real_t v = std::abs(at(A, row, col));
      if (v > max_abs) {
        max_abs = v;
        pivot = row;
      }
    }
    if (max_abs < 1e-30) {
      throw std::runtime_error("PyramidBasis: singular Vandermonde");
    }
    if (pivot != col) {
      for (int j = 0; j < n; ++j) {
        std::swap(at(A, pivot, j), at(A, col, j));
        std::swap(at(inv, pivot, j), at(inv, col, j));
      }
    }

    const real_t diag = at(A, col, col);
    const real_t inv_diag = 1.0 / diag;
    for (int j = 0; j < n; ++j) {
      at(A, col, j) *= inv_diag;
      at(inv, col, j) *= inv_diag;
    }

    for (int row = 0; row < n; ++row) {
      if (row == col) continue;
      const real_t f = at(A, row, col);
      if (std::abs(f) < 1e-30) continue;
      for (int j = 0; j < n; ++j) {
        at(A, row, j) -= f * at(A, col, j);
        at(inv, row, j) -= f * at(inv, col, j);
      }
    }
  }
}

struct PyramidBasisEntry {
  int order = 1;
  size_t n = 0;
  std::vector<BasisIndex> basis;
  std::vector<real_t> inv; // row-major [n*n] = V^{-1}
  bool ok = false;
};

class PyramidBasisCache {
public:
  static PyramidBasisCache& instance() {
    static PyramidBasisCache cache;
    return cache;
  }

  const PyramidBasisEntry& get(int p) {
    auto it = entries_.find(p);
    if (it != entries_.end()) return it->second;
    PyramidBasisEntry e;
    e.order = p;
    build_entry(e);
    auto [ins_it, _] = entries_.emplace(p, std::move(e));
    return ins_it->second;
  }

private:
  std::unordered_map<int, PyramidBasisEntry> entries_;

  static void build_entry(PyramidBasisEntry& e) {
    const int p = e.order;
    const size_t n = pyramid_lagrange_node_count(p);
    if (n == 0) {
      e.ok = false;
      return;
    }
    e.n = n;
    e.basis = build_basis_indices(p);

    const auto nodes = pyramid_reference_nodes_vtk(p);
    if (nodes.size() != n) {
      throw std::runtime_error("PyramidBasis: unexpected node count");
    }

    // Build Vandermonde V[i,j] = phi_j(node_i)
    const int N = static_cast<int>(n);
    std::vector<real_t> V(static_cast<size_t>(N) * static_cast<size_t>(N), real_t(0));
    auto v_at = [N](std::vector<real_t>& M, int r, int c) -> real_t& { return M[static_cast<size_t>(r * N + c)]; };

    std::vector<real_t> phi;
    std::vector<std::array<real_t, 3>> dphi;
    for (int i = 0; i < N; ++i) {
      eval_bcd_basis(p, e.basis, nodes[static_cast<size_t>(i)], phi, dphi);
      if (phi.size() != n) {
        throw std::runtime_error("PyramidBasis: basis evaluation size mismatch");
      }
      for (int j = 0; j < N; ++j) {
        v_at(V, i, j) = phi[static_cast<size_t>(j)];
      }
    }

#ifdef MESH_HAS_EIGEN
    Eigen::MatrixXd Vm(N, N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        Vm(i, j) = V[static_cast<size_t>(i * N + j)];
      }
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Vm);
    if (qr.rank() != N) {
      throw std::runtime_error("PyramidBasis: Vandermonde is rank-deficient");
    }
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);
    const Eigen::MatrixXd Vinv = qr.solve(I);

    e.inv.assign(static_cast<size_t>(N) * static_cast<size_t>(N), real_t(0));
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        e.inv[static_cast<size_t>(i * N + j)] = static_cast<real_t>(Vinv(i, j));
      }
    }
#else
    std::vector<real_t> Vinv;
    invert_square_matrix(V, Vinv, N);
    e.inv = std::move(Vinv);
#endif

    e.ok = true;
  }
};

} // namespace

void PyramidBasis::eval_lagrange(int p,
                                 const ParametricPoint& xi,
                                 std::vector<real_t>& N,
                                 std::vector<std::array<real_t, 3>>& dN_dxi) {
  const auto& entry = PyramidBasisCache::instance().get(p);
  if (!entry.ok || entry.n == 0) {
    throw std::runtime_error("PyramidBasis: cache not initialized");
  }

  std::vector<real_t> phi;
  std::vector<std::array<real_t, 3>> dphi;
  eval_bcd_basis(p, entry.basis, xi, phi, dphi);

  const int n = static_cast<int>(entry.n);
  if (static_cast<int>(phi.size()) != n || static_cast<int>(dphi.size()) != n) {
    throw std::runtime_error("PyramidBasis: basis evaluation size mismatch");
  }
  if (static_cast<int>(entry.inv.size()) != n * n) {
    throw std::runtime_error("PyramidBasis: inverse size mismatch");
  }

  N.assign(entry.n, 0.0);
  dN_dxi.assign(entry.n, {0, 0, 0});

  auto inv_at = [n](const std::vector<real_t>& M, int r, int c) -> real_t {
    return M[static_cast<size_t>(r * n + c)];
  };

  // N_i = sum_j inv[j,i] * phi_j
  for (int i = 0; i < n; ++i) {
    real_t Ni = 0.0;
    std::array<real_t, 3> g{0, 0, 0};
    for (int j = 0; j < n; ++j) {
      const real_t aij = inv_at(entry.inv, j, i);
      Ni += aij * phi[static_cast<size_t>(j)];
      g[0] += aij * dphi[static_cast<size_t>(j)][0];
      g[1] += aij * dphi[static_cast<size_t>(j)][1];
      g[2] += aij * dphi[static_cast<size_t>(j)][2];
    }
    N[static_cast<size_t>(i)] = Ni;
    dN_dxi[static_cast<size_t>(i)] = g;
  }
}

} // namespace svmp

