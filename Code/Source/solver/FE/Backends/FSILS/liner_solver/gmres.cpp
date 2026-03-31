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

//-------------------------------------------------------------------------
// Generalized minimum residual algorithm.
// Optimized for VMS consistent tangents using CGS2 (Re-orthogonalization),
// Cache-Blocked Fused BLAS updates, zero-allocation tracking, and hypot stability.
//-------------------------------------------------------------------------

#include "gmres.h"

#include "fsils_api.hpp"

#include "add_bc_mul.h"
#include "bcast.h"
#include "dot.h"
#include "norm.h"
#include "omp_la.h"
#include "spar_mul.h"

#include "Array3.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <vector>
#include <algorithm>
#include <string>
#include <cctype>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gmres {

namespace {

using fe_fsi_linear_solver::fsils_int;

constexpr fsils_int BLOCK_SIZE = 256;
constexpr int DOT_THREAD_PAD = 8; // doubles (64 bytes) to reduce false sharing

[[nodiscard]] int max_omp_threads() noexcept
{
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

[[nodiscard]] int dot_thread_stride(const int max_vecs) noexcept
{
  return max_vecs + DOT_THREAD_PAD;
}

[[nodiscard]] int omp_thread_num() noexcept
{
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

[[nodiscard]] bool use_serial_hot_path(const int num_threads) noexcept
{
  return num_threads <= 1;
}

[[nodiscard]] std::string to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

[[nodiscard]] bool parse_bool_env(const char* name, bool default_value) noexcept
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  std::string v(env);
  v = to_lower(v);
  if (v.empty()) {
    return default_value;
  }
  if (v == "1" || v == "true" || v == "on" || v == "yes") {
    return true;
  }
  if (v == "0" || v == "false" || v == "off" || v == "no") {
    return false;
  }
  return default_value;
}

[[nodiscard]] int parse_int_env(const char* name, int default_value) noexcept
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  try {
    return std::stoi(env);
  } catch (...) {
    return default_value;
  }
}

[[nodiscard]] double parse_double_env(const char* name, double default_value) noexcept
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  try {
    return std::stod(env);
  } catch (...) {
    return default_value;
  }
}

struct GmresEnhancements {
  enum class RecycleUpdateMode { each_restart, once_per_solve, off };
  enum class ReorthMode { selective, off };
  int recycle_k = 0;
  RecycleUpdateMode recycle_update_mode = RecycleUpdateMode::each_restart;
  bool recycle_adaptive_keep_drop = false;
  double recycle_drop_tol = 1e-3;       // normalized coefficient threshold
  double recycle_drop_rel = 0.05;       // relative-to-max threshold
  double recycle_score_alpha = 0.8;     // EWMA smoothing
  int recycle_drop_window = 3;          // consecutive "low score" restarts to drop
  int recycle_drop_min_k = 0;           // never drop below this
  // When adaptive keep/drop is enabled and the recycle space is full, the update
  // needs to "make room" for new harmonic Ritz vectors. This controls how many
  // low-score recycle vectors are replaced each update:
  //   -1: disable replacement (freeze when full)
  //    0: auto (replace ~k_req/2)
  //   >0: replace exactly this many
  int recycle_replace_n = 0;
  bool use_mgs_dgks = false;
  double mgs_dgks_ratio = 0.5;
  ReorthMode reorth_mode = ReorthMode::selective;  // default: current behavior
  bool verbose = false;
};

[[nodiscard]] const GmresEnhancements& gmres_enhancements()
{
  static const GmresEnhancements cfg = []() {
    GmresEnhancements c{};
    c.recycle_k = std::max(0, parse_int_env("SVMP_FSILS_GMRES_RECYCLE_K", 0));
    c.verbose = parse_bool_env("SVMP_FSILS_GMRES_RECYCLE_VERBOSE", false);

    const char* update = std::getenv("SVMP_FSILS_GMRES_RECYCLE_UPDATE");
    if (update) {
      std::string v = to_lower(std::string(update));
      if (v == "off" || v == "no" || v == "false" || v == "0") {
        c.recycle_update_mode = GmresEnhancements::RecycleUpdateMode::off;
      } else if (v == "once" || v == "solve") {
        c.recycle_update_mode = GmresEnhancements::RecycleUpdateMode::once_per_solve;
      } else if (v == "each" || v == "restart" || v == "restarts" || v == "on" || v == "true" || v == "1") {
        c.recycle_update_mode = GmresEnhancements::RecycleUpdateMode::each_restart;
      }
    }

    c.recycle_adaptive_keep_drop = parse_bool_env("SVMP_FSILS_GMRES_RECYCLE_ADAPTIVE", false);
    c.recycle_drop_tol = parse_double_env("SVMP_FSILS_GMRES_RECYCLE_DROP_TOL", 1e-3);
    if (!(c.recycle_drop_tol >= 0.0 && c.recycle_drop_tol <= 1.0)) {
      c.recycle_drop_tol = 1e-3;
    }
    c.recycle_drop_rel = parse_double_env("SVMP_FSILS_GMRES_RECYCLE_DROP_REL", 0.05);
    if (!(c.recycle_drop_rel >= 0.0 && c.recycle_drop_rel <= 1.0)) {
      c.recycle_drop_rel = 0.05;
    }
    c.recycle_score_alpha = parse_double_env("SVMP_FSILS_GMRES_RECYCLE_SCORE_ALPHA", 0.8);
    if (!(c.recycle_score_alpha >= 0.0 && c.recycle_score_alpha <= 1.0)) {
      c.recycle_score_alpha = 0.8;
    }
    c.recycle_drop_window = std::max(1, parse_int_env("SVMP_FSILS_GMRES_RECYCLE_DROP_WINDOW", 3));
    c.recycle_drop_min_k = std::max(0, parse_int_env("SVMP_FSILS_GMRES_RECYCLE_DROP_MIN_K", 0));
    c.recycle_replace_n = parse_int_env("SVMP_FSILS_GMRES_RECYCLE_REPLACE_N", 0);
    if (c.recycle_replace_n < -1) {
      c.recycle_replace_n = -1;
    }

    const char* orth = std::getenv("SVMP_FSILS_GMRES_ORTH");
    if (orth) {
      std::string v = to_lower(std::string(orth));
      if (v == "mgs" || v == "modified" || v == "modified-gram-schmidt") {
        c.use_mgs_dgks = true;
      }
    }

    c.mgs_dgks_ratio = parse_double_env("SVMP_FSILS_GMRES_DGKS_RATIO", 0.5);
    if (!(c.mgs_dgks_ratio > 0.0 && c.mgs_dgks_ratio < 1.0)) {
      c.mgs_dgks_ratio = 0.5;
    }

    const char* reorth_env = std::getenv("SVMP_FSILS_GMRES_REORTH");
    if (reorth_env) {
      std::string v = to_lower(std::string(reorth_env));
      if (v == "off" || v == "no" || v == "false" || v == "0" || v == "none") {
        c.reorth_mode = GmresEnhancements::ReorthMode::off;
      }
    }

    return c;
  }();
  return cfg;
}

[[nodiscard]] Eigen::MatrixXd harmonic_ritz_coeffs(const Array<double>& h, const int m, const int k_target)
{
  if (m <= 0 || k_target <= 0) {
    return Eigen::MatrixXd{};
  }

  // Build H_m from the leading m-by-m block of the (m+1)-by-m Hessenberg Hbar.
  Eigen::MatrixXd Hm = Eigen::MatrixXd::Zero(m, m);
  for (int col = 0; col < m; ++col) {
    for (int row = 0; row < m; ++row) {
      Hm(row, col) = h(row, col);
    }
  }

  // Harmonic Ritz matrix (Morgan 2002):
  //   Hhat = Hm + h_{m+1,m}^2 * f * e_m^T, where Hm^T f = e_m.
  // When h_{m+1,m} ~ 0 (happy breakdown), fall back to standard Ritz on Hm.
  double h_extra = 0.0;
  if (m >= 2) {
    h_extra = h(m, m - 1);
  }

  Eigen::MatrixXd Hhat = Hm;
  const double extra2 = h_extra * h_extra;
  if (m >= 2 && std::isfinite(extra2) && extra2 > 0.0) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(m);
    e(m - 1) = 1.0;
    Eigen::VectorXd f = Hm.transpose().fullPivLu().solve(e);
    if (f.allFinite()) {
      Hhat.col(m - 1) += extra2 * f;
    }
  }

  Eigen::ComplexEigenSolver<Eigen::MatrixXd> ces;
  ces.compute(Hhat);
  if (ces.info() != Eigen::Success) {
    return Eigen::MatrixXd{};
  }

  const auto evals = ces.eigenvalues();
  const auto evecs = ces.eigenvectors(); // columns

  std::vector<int> idx(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) idx[static_cast<std::size_t>(i)] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return std::abs(evals[a]) < std::abs(evals[b]);
  });

  // Convert complex eigenvectors to a set of real coefficient vectors.
  // Use only one vector from each conjugate pair (imag >= 0), and split
  // into real/imag parts when needed.
  constexpr double imag_pair_tol = 1e-12;
  std::vector<Eigen::VectorXd> candidates;
  candidates.reserve(static_cast<std::size_t>(k_target));
  for (const int id : idx) {
    if (static_cast<int>(candidates.size()) >= k_target) {
      break;
    }
    const std::complex<double> lam = evals[id];
    if (lam.imag() < -imag_pair_tol) {
      continue; // handle the conjugate with +imag
    }

    const Eigen::VectorXcd v = evecs.col(id);
    const double imag_norm = v.imag().norm();
    if (imag_norm <= imag_pair_tol) {
      Eigen::VectorXd vr = v.real();
      if (vr.norm() > 0.0 && vr.allFinite()) {
        candidates.push_back(std::move(vr));
      }
      continue;
    }

    Eigen::VectorXd vr = v.real();
    Eigen::VectorXd vi = v.imag();
    if (vr.norm() > 0.0 && vr.allFinite()) {
      candidates.push_back(std::move(vr));
      if (static_cast<int>(candidates.size()) >= k_target) {
        break;
      }
    }
    if (vi.norm() > 0.0 && vi.allFinite()) {
      candidates.push_back(std::move(vi));
    }
  }

  if (candidates.empty()) {
    return Eigen::MatrixXd{};
  }

  // Orthonormalize candidates in coefficient space (MGS).
  const int k_max = std::min(k_target, static_cast<int>(candidates.size()));
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(m, k_max);
  int kept = 0;
  constexpr double drop_tol = 1e-14;
  for (int c = 0; c < k_max; ++c) {
    Eigen::VectorXd v = candidates[static_cast<std::size_t>(c)];
    for (int j = 0; j < kept; ++j) {
      const double a = Q.col(j).dot(v);
      v.noalias() -= a * Q.col(j);
    }
    const double nrm = v.norm();
    if (!(nrm > drop_tol) || !std::isfinite(nrm)) {
      continue;
    }
    Q.col(kept) = v / nrm;
    kept++;
  }

  if (kept <= 0) {
    return Eigen::MatrixXd{};
  }
  return Q.leftCols(kept);
}

// =======================================================================
// FUSED CACHE-BLOCKED VECTOR KERNELS (GEMV-like Operations)
// Reduces O(i) DRAM passes over the grid to O(1) during Gram-Schmidt
// by operating on L1-sized tiles and fusing per-basis-vector work.
// =======================================================================

template <int DOF>
void fused_dot_v_impl(const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                      std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[DOF * BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;
      const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

      const double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] = v_block[idx];
      }

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          sum += uj_block[idx] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }
}

void fused_dot_v_dyn(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                     std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj = u.slice_data(j);
        double sum = 0.0;
        for (fsils_int k = k_start; k < k_end; ++k) {
          const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
          double tmp = 0.0;
          for (int d = 0; d < dof; ++d) {
            tmp += uj[base + static_cast<size_t>(d)] * v[base + static_cast<size_t>(d)];
          }
          sum += tmp;
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }
}

void fused_dot_v(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                 std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  switch (dof) {
    case 1: fused_dot_v_impl<1>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    case 2: fused_dot_v_impl<2>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    case 3: fused_dot_v_impl<3>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    case 4: fused_dot_v_impl<4>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    default: fused_dot_v_dyn(dof, mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
  }
}

template <int DOF>
double fused_dot_zz_v_impl(const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                           std::vector<double>& h_col, double* thread_buf, const int thread_stride,
                           const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  if (use_serial_hot_path(num_threads)) {
    std::fill(h_col.begin(), h_col.begin() + num_vecs, 0.0);
    double zz_sum = 0.0;
    alignas(64) double vbuf[DOF * BLOCK_SIZE];

    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;
      const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

      const double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] = v_block[idx];
      }

      double zz = 0.0;
      #pragma omp simd reduction(+:zz)
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        zz += vbuf[idx] * vbuf[idx];
      }
      zz_sum += zz;

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          sum += uj_block[idx] * vbuf[idx];
        }
        h_col[j] += sum;
      }
    }

    return zz_sum;
  }

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[DOF * BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;
      const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

      const double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] = v_block[idx];
      }

      double zz = 0.0;
      #pragma omp simd reduction(+:zz)
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        zz += vbuf[idx] * vbuf[idx];
      }
      local[num_vecs] += zz;

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          sum += uj_block[idx] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }

  double zz_sum = 0.0;
  for (int t = 0; t < num_threads; ++t) {
    zz_sum += thread_buf[static_cast<size_t>(t) * thread_stride + num_vecs];
  }
  return zz_sum;
}

double fused_dot_zz_v_dyn(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                          std::vector<double>& h_col, double* thread_buf, const int thread_stride,
                          const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  if (use_serial_hot_path(num_threads)) {
    std::fill(h_col.begin(), h_col.begin() + num_vecs, 0.0);
    double zz_sum = 0.0;

    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj = u.slice_data(j);
        double sum = 0.0;
        for (fsils_int k = k_start; k < k_end; ++k) {
          const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
          double tmp = 0.0;
          for (int d = 0; d < dof; ++d) {
            tmp += uj[base + static_cast<size_t>(d)] * v[base + static_cast<size_t>(d)];
          }
          sum += tmp;
        }
        h_col[j] += sum;
      }

      double zz = 0.0;
      for (fsils_int k = k_start; k < k_end; ++k) {
        const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
        for (int d = 0; d < dof; ++d) {
          const double val = v[base + static_cast<size_t>(d)];
          zz += val * val;
        }
      }
      zz_sum += zz;
    }

    return zz_sum;
  }

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj = u.slice_data(j);
        double sum = 0.0;
        for (fsils_int k = k_start; k < k_end; ++k) {
          const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
          double tmp = 0.0;
          for (int d = 0; d < dof; ++d) {
            tmp += uj[base + static_cast<size_t>(d)] * v[base + static_cast<size_t>(d)];
          }
          sum += tmp;
        }
        local[j] += sum;
      }

      double zz = 0.0;
      for (fsils_int k = k_start; k < k_end; ++k) {
        const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
        for (int d = 0; d < dof; ++d) {
          const double val = v[base + static_cast<size_t>(d)];
          zz += val * val;
        }
      }
      local[num_vecs] += zz;
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }

  double zz_sum = 0.0;
  for (int t = 0; t < num_threads; ++t) {
    zz_sum += thread_buf[static_cast<size_t>(t) * thread_stride + num_vecs];
  }
  return zz_sum;
}

double fused_dot_zz_v(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                      std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  switch (dof) {
    case 1: return fused_dot_zz_v_impl<1>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    case 2: return fused_dot_zz_v_impl<2>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    case 3: return fused_dot_zz_v_impl<3>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    case 4: return fused_dot_zz_v_impl<4>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    default: return fused_dot_zz_v_dyn(dof, mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
  }
}

template <int DOF>
double fused_update_norm_v_impl(const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                                Array3<double>& u, const int i, Array<double>& u_next, const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();
  double local_sq = 0.0;

  // Deterministic partial-sum pattern: each thread stores its local sum
  // in a fixed slot, then partials are summed in thread-ID order.
  {
#ifdef _OPENMP
    const int max_t = omp_get_max_threads();
    constexpr int kStackMax = 64;
    double stack_partials[kStackMax];
    double* partials = (max_t <= kStackMax) ? stack_partials : new double[max_t];
    for (int t = 0; t < max_t; ++t) partials[t] = 0.0;

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      double thread_sq = 0.0;
      #pragma omp for schedule(static)
#else
    {
      double thread_sq = 0.0;
#endif
      for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
        const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
        const fsils_int blk_nodes = k_end - k_start;
        const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

        alignas(64) double vbuf[DOF * BLOCK_SIZE];
        double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

        #pragma omp simd
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          vbuf[idx] = v_block[idx];
        }

        for (int j = 0; j < num_vecs; ++j) {
          const double hj = h_factors[j];
          if (hj == 0.0) {
            continue;
          }

          const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
          #pragma omp simd
          for (size_t idx = 0; idx < blk_elems; ++idx) {
            vbuf[idx] -= hj * uj_block[idx];
          }
        }

        fsils_int owned_nodes = 0;
        if (k_start < mynNo) {
          owned_nodes = std::min(k_end, mynNo) - k_start;
        }

        const size_t owned_elems = static_cast<size_t>(owned_nodes) * static_cast<size_t>(DOF);
        double blk_sq = 0.0;
        #pragma omp simd reduction(+:blk_sq)
        for (size_t idx = 0; idx < owned_elems; ++idx) {
          blk_sq += vbuf[idx] * vbuf[idx];
        }
        thread_sq += blk_sq;

        #pragma omp simd
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          v_block[idx] = vbuf[idx];
        }
      }
#ifdef _OPENMP
      partials[tid] = thread_sq;
    } // end omp parallel
    for (int t = 0; t < max_t; ++t) local_sq += partials[t];
    if (max_t > kStackMax) delete[] partials;
#else
      local_sq = thread_sq;
    }
#endif
  }

  double global_sq = local_sq;
  if (commu.nTasks != 1) {
    double tmp = 0.0;
    fsils_allreduce_sum(&local_sq, &tmp, 1, cm_mod::mpreal, commu);
    global_sq = tmp;
  }
  return std::sqrt(global_sq);
}

double fused_update_norm_v_dyn(const int dof, const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                               Array3<double>& u, const int i, Array<double>& u_next, const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      double* vv = v + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vv[idx] -= hj * uj[idx];
      }
    }
  }

  return norm::fsi_ls_normv(dof, mynNo, commu, u_next);
}

double fused_update_norm_v(const int dof, const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                           Array3<double>& u, const int i, Array<double>& u_next, const std::vector<double>& h_factors)
{
  switch (dof) {
    case 1: return fused_update_norm_v_impl<1>(nNo, mynNo, commu, u, i, u_next, h_factors);
    case 2: return fused_update_norm_v_impl<2>(nNo, mynNo, commu, u, i, u_next, h_factors);
    case 3: return fused_update_norm_v_impl<3>(nNo, mynNo, commu, u, i, u_next, h_factors);
    case 4: return fused_update_norm_v_impl<4>(nNo, mynNo, commu, u, i, u_next, h_factors);
    default: return fused_update_norm_v_dyn(dof, nNo, mynNo, commu, u, i, u_next, h_factors);
  }
}

template <int DOF>
void fused_update_v_inplace_impl(const fsils_int nNo, Array3<double>& u, const int i, Array<double>& u_next,
                                 const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  if (use_serial_hot_path(max_omp_threads())) {
    for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
      const fsils_int blk_nodes = k_end - k_start;
      const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

      alignas(64) double vbuf[DOF * BLOCK_SIZE];
      double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] = v_block[idx];
      }

      for (int j = 0; j < num_vecs; ++j) {
        const double hj = h_factors[j];
        if (hj == 0.0) {
          continue;
        }

        const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
        #pragma omp simd
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          vbuf[idx] -= hj * uj_block[idx];
        }
      }

      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        v_block[idx] = vbuf[idx];
      }
    }
    return;
  }

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;
    const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

    alignas(64) double vbuf[DOF * BLOCK_SIZE];
    double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      vbuf[idx] = v_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] -= hj * uj_block[idx];
      }
    }

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      v_block[idx] = vbuf[idx];
    }
  }
}

void fused_update_v_inplace_dyn(const int dof, const fsils_int nNo, Array3<double>& u, const int i, Array<double>& u_next,
                                const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  if (use_serial_hot_path(max_omp_threads())) {
    for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
      const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
      const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

      for (int j = 0; j < num_vecs; ++j) {
        const double hj = h_factors[j];
        if (hj == 0.0) {
          continue;
        }

        const double* uj = u.slice_data(j) + block_offset;
        double* vv = v + block_offset;
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          vv[idx] -= hj * uj[idx];
        }
      }
    }
    return;
  }

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      double* vv = v + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vv[idx] -= hj * uj[idx];
      }
    }
  }
}

void fused_update_v_inplace(const int dof, const fsils_int nNo, Array3<double>& u, const int i, Array<double>& u_next,
                            const std::vector<double>& h_factors)
{
  switch (dof) {
    case 1: fused_update_v_inplace_impl<1>(nNo, u, i, u_next, h_factors); break;
    case 2: fused_update_v_inplace_impl<2>(nNo, u, i, u_next, h_factors); break;
    case 3: fused_update_v_inplace_impl<3>(nNo, u, i, u_next, h_factors); break;
    case 4: fused_update_v_inplace_impl<4>(nNo, u, i, u_next, h_factors); break;
    default: fused_update_v_inplace_dyn(dof, nNo, u, i, u_next, h_factors); break;
  }
}

template <int DOF>
void fused_copy_update_v_inplace_impl(const fsils_int nNo, const double* src, Array3<double>& u, const int i, Array<double>& u_next,
                                      const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;
    const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

    alignas(64) double vbuf[DOF * BLOCK_SIZE];
    const double* src_block = src + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      vbuf[idx] = src_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] -= hj * uj_block[idx];
      }
    }

    double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      v_block[idx] = vbuf[idx];
    }
  }
}

void fused_copy_update_v_inplace_dyn(const int dof, const fsils_int nNo, const double* src, Array3<double>& u, const int i, Array<double>& u_next,
                                     const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    const double* src_block = src + block_offset;
    double* vv = v + block_offset;
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      vv[idx] = src_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vv[idx] -= hj * uj[idx];
      }
    }
  }
}

void fused_copy_update_v_inplace(const int dof, const fsils_int nNo, const double* src, Array3<double>& u, const int i, Array<double>& u_next,
                                 const std::vector<double>& h_factors)
{
  switch (dof) {
    case 1: fused_copy_update_v_inplace_impl<1>(nNo, src, u, i, u_next, h_factors); break;
    case 2: fused_copy_update_v_inplace_impl<2>(nNo, src, u, i, u_next, h_factors); break;
    case 3: fused_copy_update_v_inplace_impl<3>(nNo, src, u, i, u_next, h_factors); break;
    case 4: fused_copy_update_v_inplace_impl<4>(nNo, src, u, i, u_next, h_factors); break;
    default: fused_copy_update_v_inplace_dyn(dof, nNo, src, u, i, u_next, h_factors); break;
  }
}

template <int DOF>
void fused_recon_v_impl(const fsils_int nNo, Array3<double>& u, const int last_i, Array<double>& X, const Vector<double>& y)
{
  const int num_vecs = last_i + 1;
  double* x = X.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;
    const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

    alignas(64) double xbuf[DOF * BLOCK_SIZE];
    double* x_block = x + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      xbuf[idx] = x_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double yj = y(j);
      if (yj == 0.0) {
        continue;
      }

      const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        xbuf[idx] += yj * uj_block[idx];
      }
    }

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      x_block[idx] = xbuf[idx];
    }
  }
}

void fused_recon_v_dyn(const int dof, const fsils_int nNo, Array3<double>& u, const int last_i, Array<double>& X, const Vector<double>& y)
{
  const int num_vecs = last_i + 1;
  double* x = X.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    for (int j = 0; j < num_vecs; ++j) {
      const double yj = y(j);
      if (yj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      double* xx = x + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        xx[idx] += yj * uj[idx];
      }
    }
  }
}

void fused_recon_v(const int dof, const fsils_int nNo, Array3<double>& u, const int last_i, Array<double>& X, const Vector<double>& y)
{
  switch (dof) {
    case 1: fused_recon_v_impl<1>(nNo, u, last_i, X, y); break;
    case 2: fused_recon_v_impl<2>(nNo, u, last_i, X, y); break;
    case 3: fused_recon_v_impl<3>(nNo, u, last_i, X, y); break;
    case 4: fused_recon_v_impl<4>(nNo, u, last_i, X, y); break;
    default: fused_recon_v_dyn(dof, nNo, u, last_i, X, y); break;
  }
}

void fused_dot_s(const fsils_int mynNo, Array<double>& u, const int i, const Vector<double>& u_next,
                 std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;

      const double* v_block = v + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] = v_block[static_cast<size_t>(idx)];
      }

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
          sum += uj_block[static_cast<size_t>(idx)] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }
}

double fused_dot_zz_s(const fsils_int mynNo, Array<double>& u, const int i, const Vector<double>& u_next,
                      std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;

      const double* v_block = v + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] = v_block[static_cast<size_t>(idx)];
      }

      double zz = 0.0;
      #pragma omp simd reduction(+:zz)
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        zz += vbuf[idx] * vbuf[idx];
      }
      local[num_vecs] += zz;

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
          sum += uj_block[static_cast<size_t>(idx)] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }

  double zz_sum = 0.0;
  for (int t = 0; t < num_threads; ++t) {
    zz_sum += thread_buf[static_cast<size_t>(t) * thread_stride + num_vecs];
  }
  return zz_sum;
}

double fused_update_norm_s(const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                           Array<double>& u, const int i, Vector<double>& u_next, const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();
  double local_sq = 0.0;

  {
#ifdef _OPENMP
    const int max_t = omp_get_max_threads();
    constexpr int kStackMax = 64;
    double stack_partials[kStackMax];
    double* partials = (max_t <= kStackMax) ? stack_partials : new double[max_t];
    for (int t = 0; t < max_t; ++t) partials[t] = 0.0;

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      double thread_sq = 0.0;
      #pragma omp for schedule(static)
#else
    {
      double thread_sq = 0.0;
#endif
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double vbuf[BLOCK_SIZE];
    double* v_block = v + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      vbuf[idx] = v_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] -= hj * uj_block[static_cast<size_t>(idx)];
      }
    }

    fsils_int owned_nodes = 0;
    if (k_start < mynNo) {
      owned_nodes = std::min(k_end, mynNo) - k_start;
    }

    double blk_sq = 0.0;
    #pragma omp simd reduction(+:blk_sq)
    for (fsils_int idx = 0; idx < owned_nodes; ++idx) {
      blk_sq += vbuf[idx] * vbuf[idx];
    }
    thread_sq += blk_sq;

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      v_block[static_cast<size_t>(idx)] = vbuf[idx];
    }
  }
#ifdef _OPENMP
      partials[tid] = thread_sq;
    } // end omp parallel
    for (int t = 0; t < max_t; ++t) local_sq += partials[t];
    if (max_t > kStackMax) delete[] partials;
#else
      local_sq = thread_sq;
    }
#endif
  }

  double global_sq = local_sq;
  if (commu.nTasks != 1) {
    double tmp = 0.0;
    fsils_allreduce_sum(&local_sq, &tmp, 1, cm_mod::mpreal, commu);
    global_sq = tmp;
  }
  return std::sqrt(global_sq);
}

void fused_update_s_inplace(const fsils_int nNo, Array<double>& u, const int i, Vector<double>& u_next,
                            const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double vbuf[BLOCK_SIZE];
    double* v_block = v + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      vbuf[idx] = v_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] -= hj * uj_block[static_cast<size_t>(idx)];
      }
    }

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      v_block[static_cast<size_t>(idx)] = vbuf[idx];
    }
  }
}

void fused_copy_update_s_inplace(const fsils_int nNo, const double* src, Array<double>& u, const int i, Vector<double>& u_next,
                                 const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double vbuf[BLOCK_SIZE];
    const double* src_block = src + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      vbuf[idx] = src_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] -= hj * uj_block[static_cast<size_t>(idx)];
      }
    }

    double* v_block = v + static_cast<size_t>(k_start);
    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      v_block[static_cast<size_t>(idx)] = vbuf[idx];
    }
  }
}

void fused_recon_s(const fsils_int nNo, Array<double>& u, const int last_i, Vector<double>& X, const Vector<double>& y)
{
  const int num_vecs = last_i + 1;
  double* x = X.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double xbuf[BLOCK_SIZE];
    double* x_block = x + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      xbuf[idx] = x_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double yj = y(j);
      if (yj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        xbuf[idx] += yj * uj_block[static_cast<size_t>(idx)];
      }
    }

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      x_block[static_cast<size_t>(idx)] = xbuf[idx];
    }
  }
}

} // namespace

/// @brief Pre-calculates the boundary condition normalization factor.
/// Removed O(N) heap allocation, using fast local accumulations.
void bc_pre(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
            const int mynNo, const int nNo)
{
  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto &face = lhs.face[faIn];

    if (face.coupledFlag) {
      // Use the face's own DOF count (set by the caller based on which
      // components participate in this coupled BC), clamped to the
      // system DOF count for safety.
      const int face_dof = std::min(face.dof, dof);

      if (face.sharedFlag) {
        double local_nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          if (Ac < mynNo) {
            for (int i = 0; i < face_dof; i++) {
              local_nS += face.valM(i,a) * face.valM(i,a);
            }
          }
        }

        double global_nS = 0.0;
        if (lhs.commu.nTasks > 1) {
          fsils_allreduce_sum(&local_nS, &global_nS, 1, cm_mod::mpreal, lhs.commu);
        } else {
          global_nS = local_nS;
        }
        face.nS = global_nS;

      } else {
        face.nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          for (int i = 0; i < face_dof; i++) {
            face.nS += face.valM(i,a) * face.valM(i,a);
          }
        }
      }
    }
  }

  add_bc_mul::compute_reduced_update_preconditioner_coupling(lhs);
}

/// @brief Solver the system Val * X = R.
///
/// Reproduces the Fortran 'GMRES' subroutine.
void gmres(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
           const Array<double>& Val, const Array<double>& R, Array<double>& X)
{
  using namespace fe_fsi_linear_solver;
  const auto& enh = gmres_enhancements();
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  const bool has_coupled_bc = std::any_of(lhs.face.begin(), lhs.face.end(),
      [](const auto& face) { return face.coupledFlag; });

  ls.ws.ensure_gmres_v(dof, nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& u = ls.ws.u3;
  auto& unCondU = ls.ws.unCondU;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  double time = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;
  double eps = 0.0;
  int last_i = 0;
  int restart_cycles = 0;
  X = 0.0;
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  // Stagnation detection (see gmres_v for rationale).
  double prev_fNorm = 0.0;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;
  const bool allow_heuristic_early_stop = !ls.exact_convergence;

  if (has_coupled_bc) {
    bc_pre(lhs, ls, dof, mynNo, nNo);
  }

  for (int l = 0; l < ls.mItr; l++) {
    restart_cycles++;
    auto u_slice = u.rslice(0);

    if (l == 0) {
      u.set_slice(0, R);
    } else {
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, X, u_slice);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, X, u_slice);
      ls.itr = ls.itr + 1;

      // Zero-allocation residual: u_slice = R - u_slice
      omp_la::omp_axpby_v(dof, nNo, u_slice, R, -1.0, u_slice);
    }

    if (has_coupled_bc) {
      auto unCondU_ref = u.rslice(0);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice);
    }

    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice);

    if (l == 0) {
      eps = err[0];
      ls.iNorm = eps;
      ls.fNorm = eps;
      eps = std::max(ls.absTol, ls.relTol*eps);
    }

    ls.dB = ls.fNorm;

    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true;
      break;
    }

    omp_la::omp_mul_v(dof, nNo, 1.0 / err[0], u_slice);

    for (int i = 0; i < ls.sD; i++) {
      last_i = i;
      auto u_slice_prev = u.rslice(i);
      auto u_slice_next = u.rslice(i+1);

      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, u_slice_prev, u_slice_next);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, u_slice_prev, u_slice_next);
      ls.itr = ls.itr + 1;

      if (has_coupled_bc) {
        auto unCondU_ref = u.rslice(i+1);
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice_next);
      }

      // ----------------------------------------------------------------------
      // Sequential CGS with Pythagorean norm trick (streaming access pattern).
      // Each dot product reads one u[j] linearly while u_slice_next stays in L2.
      // ----------------------------------------------------------------------
      for (int j = 0; j <= i; j++) {
        h_col[j] = dot::fsils_nc_dot_v(dof, mynNo, u.rslice(j), u_slice_next);
      }
      const double zz_local = dot::fsils_nc_dot_v(dof, mynNo, u_slice_next, u_slice_next);
      h_col[static_cast<size_t>(i+1)] = zz_local;
      bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      const double old_norm_sq = h_col[static_cast<size_t>(i+1)];

      for (int j = 0; j <= i; j++) {
        if (h_col[j] != 0.0) {
          omp_la::omp_sum_v(dof, nNo, -h_col[j], u_slice_next, u.rslice(j));
        }
      }

      double new_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice_next);
      const double new_norm_sq = new_norm * new_norm;

      // CGS2: Selective reorthogonalization (Pythagorean check)
      bool do_reorth = false;
      if (!ls.disable_reorth &&
          enh.reorth_mode != GmresEnhancements::ReorthMode::off &&
          std::isfinite(new_norm_sq)) {
        if (enh.use_mgs_dgks) {
          const double ratio2 = enh.mgs_dgks_ratio * enh.mgs_dgks_ratio;
          do_reorth = new_norm_sq < ratio2 * old_norm_sq;
        } else {
          do_reorth = new_norm_sq < proj_sq_norm;
        }
      }

      if (do_reorth) {
        for (int j = 0; j <= i; j++) {
          h_col[j] = dot::fsils_nc_dot_v(dof, mynNo, u.rslice(j), u_slice_next);
        }
        const double zz2_local = dot::fsils_nc_dot_v(dof, mynNo, u_slice_next, u_slice_next);
        h_col[static_cast<size_t>(i+1)] = zz2_local;
        bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

        for (int j = 0; j <= i; ++j) {
          h(j,i) += h_col[j];
        }

        for (int j = 0; j <= i; j++) {
          if (h_col[j] != 0.0) {
            omp_la::omp_sum_v(dof, nNo, -h_col[j], u_slice_next, u.rslice(j));
          }
        }
        new_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice_next);
      }
      h(i+1,i) = new_norm;

      // Happy Breakdown Protection & Safe Givens Rotation
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (h(i+1,i) > breakdown_tol) {
        omp_la::omp_mul_v(dof, nNo, 1.0/h(i+1,i), u_slice_next);
      } else {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0; s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot; s(i) = h(i+1,i) / tmp_hypot;
      }

      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;

      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);

      if (std::abs(err(i+1)) < eps || breakdown) {
        ls.suc = true;
        break;
      }
    }

    if (last_i >= ls.sD) last_i = ls.sD - 1;

    for (int i = 0; i <= last_i; i++) y(i) = err(i);

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) y(j) -= h(j,k)*y(k);
      y(j) /= h(j,j);
    }

    fused_recon_v(dof, nNo, u, last_i, X, y);

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;

    // Stagnation detection (see gmres_v for rationale).
    if (allow_heuristic_early_stop && prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (allow_heuristic_early_stop && stagnation_count >= max_stagnation_restarts) {
      break;
    }
  }

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - time + ls.callD;
  ls.dB  = 10.0 * std::log(ls.fNorm / ls.dB);
  ls.stats.record_call(ls.itr - itr_before,
                       restart_cycles,
                       fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);
}

//---------
// gmres_s
//---------
// Reproduces the Fortran 'GMRESS' subroutine.
//
void gmres_s(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
             const Vector<double>& Val, Vector<double>& R)
{
  using namespace fe_fsi_linear_solver;
  const auto& enh = gmres_enhancements();
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_gmres_s(nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& u = ls.ws.u2;
  auto& X = ls.ws.Xs;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;

  double eps = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol*eps);
  ls.itr = 0;
  int last_i = 0;
  int restart_cycles = 0;
  X = 0.0;

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    ls.stats.record_call(ls.itr - itr_before,
                         restart_cycles,
                         fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                         /*setup_seconds=*/0.0,
                         ls.callD - callD_before);
    return;
  }

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  // Stagnation detection (see gmres_v for rationale).
  double prev_fNorm = ls.fNorm;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;
  const bool allow_heuristic_early_stop = !ls.exact_convergence;

  for (int l = 0; l < ls.mItr; l++) {
    restart_cycles++;
    ls.dB = ls.fNorm;
    ls.itr++;

    auto u_col_curr = u.rcol(0);

    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, X, u_col_curr);
    omp_la::omp_axpby_s(nNo, u_col_curr, R, -1.0, u_col_curr);

    err[0] = norm::fsi_ls_norms(mynNo, lhs.commu, u_col_curr);
    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true; break;
    }

    omp_la::omp_mul_s(nNo, 1.0 / err[0], u_col_curr);

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;

      auto u_col_prev = u.rcol(i);
      auto u_col_next = u.rcol(i+1);
      spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, u_col_prev, u_col_next);

      // --- CGS Step 1 with Pythagorean trick ---
      const double zz_local = fused_dot_zz_s(mynNo, u, i, u_col_next, h_col,
                                              dot_thread.data(), thread_stride, num_threads);
      h_col[static_cast<size_t>(i+1)] = zz_local;
      bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      const double old_norm_sq = h_col[static_cast<size_t>(i+1)];

      double new_norm = fused_update_norm_s(nNo, mynNo, lhs.commu, u, i, u_col_next, h_col);
      const double new_norm_sq = new_norm * new_norm;

      // CGS2: Selective reorthogonalization (Pythagorean check)
      bool do_reorth = false;
      if (!ls.disable_reorth &&
          enh.reorth_mode != GmresEnhancements::ReorthMode::off &&
          std::isfinite(new_norm_sq)) {
        if (enh.use_mgs_dgks) {
          const double ratio2 = enh.mgs_dgks_ratio * enh.mgs_dgks_ratio;
          do_reorth = new_norm_sq < ratio2 * old_norm_sq;
        } else {
          do_reorth = new_norm_sq < proj_sq_norm;
        }
      }

      if (do_reorth) {
        const double zz2_local = fused_dot_zz_s(mynNo, u, i, u_col_next, h_col,
                                                  dot_thread.data(), thread_stride, num_threads);
        h_col[static_cast<size_t>(i+1)] = zz2_local;
        bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

        for (int j = 0; j <= i; ++j) {
          h(j,i) += h_col[j];
        }

        new_norm = fused_update_norm_s(nNo, mynNo, lhs.commu, u, i, u_col_next, h_col);
      }
      h(i+1,i) = new_norm;

      // Happy Breakdown Check
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (h(i+1,i) > breakdown_tol) {
        omp_la::omp_mul_s(nNo, 1.0/h(i+1,i), u_col_next);
      } else {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0; s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot; s(i) = h(i+1,i) / tmp_hypot;
      }

      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;
      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);

      if (std::abs(err(i+1)) < eps || breakdown) {
        ls.suc = true;
        break;
      }
    }

    if (last_i >= ls.sD) last_i = ls.sD - 1;

    for (int i = 0; i <= last_i; i++) y(i) = err(i);

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) y(j) -= h(j,k)*y(k);
      y(j) /= h(j,j);
    }

    fused_recon_s(nNo, u, last_i, X, y);

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;

    // Stagnation detection (see gmres_v for rationale).
    if (allow_heuristic_early_stop && prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (allow_heuristic_early_stop && stagnation_count >= max_stagnation_restarts) {
      break;
    }
  }

  R = X;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB  = 10.0 * std::log(ls.fNorm / ls.dB);
  ls.stats.record_call(ls.itr - itr_before,
                       restart_cycles,
                       fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);
}

//---------
// gmres_v
//---------
// Generalized minimum residual algorithm implemented for vector problems.
//
// Reproduces the Fortran 'GMRESV' subroutine.
//
void gmres_v(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
             const Array<double>& Val, Array<double>& R)
{
  using namespace fe_fsi_linear_solver;
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  const bool has_coupled_bc = std::any_of(lhs.face.begin(), lhs.face.end(),
      [](const auto& face) { return face.coupledFlag; });

  ls.ws.ensure_gmres_v(dof, nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& u = ls.ws.u3;
  auto& X = ls.ws.X2;
  auto& unCondU = ls.ws.unCondU;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;

  // ===== TIMING PROFILE =====
  double tp_spmv = 0.0, tp_bc_mul = 0.0, tp_dot_gs = 0.0;
  double tp_allreduce = 0.0, tp_gs_update = 0.0, tp_norm = 0.0;
  double tp_vecops = 0.0, tp_givens = 0.0, tp_recon = 0.0;
  int tp_reorth_count = 0, tp_restarts = 0;
  auto TP = [](){ return fe_fsi_linear_solver::fsils_cpu_t(); };
  double tp0;
  // ===========================

  tp0 = TP();
  double eps = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  tp_norm += TP() - tp0;

  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol*eps);
  ls.itr = 0;
  int last_i = 0;
  int restart_cycles = 0;
  X = 0.0;

  bc_pre(lhs, ls, dof, mynNo, nNo);

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    ls.stats.record_call(ls.itr - itr_before,
                         restart_cycles,
                         fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                         /*setup_seconds=*/0.0,
                         ls.callD - callD_before);
    return;
  }

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  // ===== RECYCLED/DEFLATED GMRES (optional) =====
  const auto& enh = gmres_enhancements();
  const int recycle_k_req = std::min(enh.recycle_k, ls.sD);
  if (recycle_k_req > 0) {
    ls.ws.ensure_recycle_v(dof, nNo, recycle_k_req);
  }
  auto& recycle_U = ls.ws.recycle_U3;
  auto& recycle_C = ls.ws.recycle_C3;
  auto& recycle_y = ls.ws.recycle_y;
  auto& recycle_score = ls.ws.recycle_score;
  auto& recycle_drop_streak = ls.ws.recycle_drop_streak;
  int recycle_k = (recycle_k_req > 0) ? std::min(ls.ws.recycle_k, recycle_k_req) : 0;
  bool recycle_updated_this_solve = false;
  std::vector<double> recycle_B;
  if (recycle_k_req > 0) {
    recycle_B.resize(static_cast<size_t>(recycle_k_req) * static_cast<size_t>(ls.sD), 0.0);
  }

  auto orthonormalize_recycle_C_inplace = [&](int k_in) -> int {
    if (k_in <= 0) {
      return 0;
    }

    // Drop near-zero vectors (relative to the initial residual norm).
    const double drop_tol = std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(ls.iNorm, 1.0);
    int kept = 0;

    for (int j = 0; j < k_in; ++j) {
      if (kept != j) {
        recycle_U.rslice(kept) = recycle_U.rslice(j);
        recycle_C.rslice(kept) = recycle_C.rslice(j);
        if (static_cast<int>(recycle_score.size()) > j) {
          recycle_score[static_cast<size_t>(kept)] = recycle_score[static_cast<size_t>(j)];
        }
        if (static_cast<int>(recycle_drop_streak.size()) > j) {
          recycle_drop_streak[static_cast<size_t>(kept)] = recycle_drop_streak[static_cast<size_t>(j)];
        }
      }

      auto cj = recycle_C.rslice(kept);
      auto uj = recycle_U.rslice(kept);

      // MGS against previous C columns; apply identical transforms to U.
      for (int i = 0; i < kept; ++i) {
        const auto ci = recycle_C.rslice(i);
        const auto ui = recycle_U.rslice(i);
        double a = dot::fsils_nc_dot_v(dof, mynNo, ci, cj);
        bcast::fsils_bcast(a, lhs.commu);
        if (a != 0.0) {
          omp_la::omp_sum_v(dof, nNo, -a, cj, ci);
          omp_la::omp_sum_v(dof, nNo, -a, uj, ui);
        }
      }

      const double nrm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, cj);
      if (!(nrm > drop_tol) || !std::isfinite(nrm)) {
        continue;
      }

      const double inv = 1.0 / nrm;
      omp_la::omp_mul_v(dof, nNo, inv, cj);
      omp_la::omp_mul_v(dof, nNo, inv, uj);
      kept++;
      if (kept >= recycle_k_req) {
        break;
      }
    }

    return kept;
  };

  auto rebuild_recycle_C_from_U = [&]() {
    if (recycle_k <= 0) {
      return;
    }

    // C := A*U for the current matrix.
    for (int j = 0; j < recycle_k; ++j) {
      const auto uj = recycle_U.rslice(j);
      auto cj = recycle_C.rslice(j);
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, uj, cj);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, uj, cj);
      if (has_coupled_bc) {
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, cj, cj);
      }
    }

    recycle_k = orthonormalize_recycle_C_inplace(recycle_k);
    ls.ws.recycle_k = recycle_k;

    if (enh.verbose && lhs.commu.task == 0) {
      std::fprintf(stderr, "[gmres_v] recycle space rebuilt: k=%d\n", recycle_k);
    }
  };

  if (recycle_k > 0) {
    rebuild_recycle_C_from_U();
  }
  // =============================================

  auto update_recycle_space_from_arnoldi = [&](const int m_iters) {
    if (recycle_k_req <= 0 || m_iters <= 0) {
      return;
    }
    // Adaptive keep/drop: keep the recycled vectors that have been "useful"
    // (based on streak/score), and refill dropped slots with new harmonic Ritz
    // candidates from the current Arnoldi basis.
    if (enh.recycle_adaptive_keep_drop && recycle_k > 0) {
      const int k = recycle_k;
      std::vector<char> keep(static_cast<size_t>(k), 0);
      int kept = 0;
      for (int j = 0; j < k; ++j) {
        const size_t jj = static_cast<size_t>(j);
        if (recycle_drop_streak[jj] < enh.recycle_drop_window) {
          keep[jj] = 1;
          kept++;
        }
      }

      const int min_k = std::min(k, enh.recycle_drop_min_k);
      if (kept < min_k && min_k > 0) {
        std::vector<double> tmp(static_cast<size_t>(k), 0.0);
        for (int j = 0; j < k; ++j) {
          tmp[static_cast<size_t>(j)] = recycle_score[static_cast<size_t>(j)];
        }
        std::nth_element(tmp.begin(), tmp.begin() + (min_k - 1), tmp.end(),
                         [](const double x, const double y) { return x > y; });
        const double cutoff = tmp[static_cast<size_t>(min_k - 1)];
        kept = 0;
        for (int j = 0; j < k; ++j) {
          const size_t jj = static_cast<size_t>(j);
          keep[jj] = (recycle_score[jj] >= cutoff) ? 1 : 0;
          if (keep[jj]) {
            kept++;
          }
        }
      }

      if (kept < k) {
        int w = 0;
        for (int j = 0; j < k; ++j) {
          if (!keep[static_cast<size_t>(j)]) {
            continue;
          }
          if (w != j) {
            recycle_U.rslice(w) = recycle_U.rslice(j);
            recycle_C.rslice(w) = recycle_C.rslice(j);
            recycle_score[static_cast<size_t>(w)] = recycle_score[static_cast<size_t>(j)];
            recycle_drop_streak[static_cast<size_t>(w)] = recycle_drop_streak[static_cast<size_t>(j)];
          }
          w++;
        }
        recycle_k = w;
        ls.ws.recycle_k = recycle_k;
      }
    } else {
      // Non-adaptive: overwrite the entire recycle space.
      recycle_k = 0;
      ls.ws.recycle_k = 0;
    }

    int slots = recycle_k_req - recycle_k;
    if (slots <= 0) {
      // Adaptive replacement: if the recycle space is already full, refresh
      // a small number of vectors (drop the lowest-score ones) so the space
      // continues to evolve across restarts (prevents "freezing" the space).
      if (enh.recycle_adaptive_keep_drop && recycle_k > 0 && enh.recycle_replace_n >= 0) {
        const int min_k = std::min(recycle_k, enh.recycle_drop_min_k);
        int replace = enh.recycle_replace_n;
        if (replace == 0) {
          replace = std::max(1, recycle_k_req / 2);
        }
        replace = std::min(replace, recycle_k - min_k);
        if (replace > 0 && static_cast<int>(recycle_score.size()) >= recycle_k) {
          std::vector<int> idx(static_cast<size_t>(recycle_k));
          for (int j = 0; j < recycle_k; ++j) {
            idx[static_cast<size_t>(j)] = j;
          }
          std::nth_element(idx.begin(), idx.begin() + replace, idx.end(), [&](const int a, const int b) {
            return recycle_score[static_cast<size_t>(a)] < recycle_score[static_cast<size_t>(b)];
          });

          std::vector<char> keep(static_cast<size_t>(recycle_k), 1);
          for (int d = 0; d < replace; ++d) {
            keep[static_cast<size_t>(idx[static_cast<size_t>(d)])] = 0;
          }

          int w = 0;
          for (int j = 0; j < recycle_k; ++j) {
            if (!keep[static_cast<size_t>(j)]) {
              continue;
            }
            if (w != j) {
              recycle_U.rslice(w) = recycle_U.rslice(j);
              recycle_C.rslice(w) = recycle_C.rslice(j);
              recycle_score[static_cast<size_t>(w)] = recycle_score[static_cast<size_t>(j)];
              recycle_drop_streak[static_cast<size_t>(w)] = recycle_drop_streak[static_cast<size_t>(j)];
            }
            w++;
          }

          recycle_k = w;
          ls.ws.recycle_k = recycle_k;
          slots = recycle_k_req - recycle_k;
          if (enh.verbose && lhs.commu.task == 0) {
            std::fprintf(stderr, "[gmres_v] recycle replace: k=%d slots=%d\n", recycle_k, slots);
          }
        }
      }

      if (slots <= 0) {
        rebuild_recycle_C_from_U();
        return;
      }
    }

    const int k_target = std::min(slots, m_iters);
    if (k_target <= 0) {
      return;
    }

    const Eigen::MatrixXd Q = harmonic_ritz_coeffs(h, m_iters, k_target);
    if (Q.size() == 0) {
      return;
    }

    const int k_add = static_cast<int>(Q.cols());
    for (int j = 0; j < k_add; ++j) {
      const int idx = recycle_k + j;
      if (static_cast<int>(recycle_score.size()) > idx) {
        recycle_score[static_cast<size_t>(idx)] = 0.0;
      }
      if (static_cast<int>(recycle_drop_streak.size()) > idx) {
        recycle_drop_streak[static_cast<size_t>(idx)] = 0;
      }
      for (int i = 0; i < m_iters; ++i) {
        y(i) = Q(i, j);
      }
      auto uj = recycle_U.rslice(idx);
      uj = 0.0;
      fused_recon_v(dof, nNo, u, m_iters - 1, uj, y);
    }

    recycle_k += k_add;
    ls.ws.recycle_k = recycle_k;

    // Rebuild C := A*U for the current operator, and orthonormalize C in-place.
    // This is more robust than forming C from the (finite-precision) Arnoldi
    // relation, and ensures the correction step uses a consistent C = A*U.
    rebuild_recycle_C_from_U();

    if (enh.verbose && lhs.commu.task == 0) {
      std::fprintf(stderr, "[gmres_v] recycle space updated: k=%d m=%d\n", recycle_k, m_iters);
    }
  };

  auto update_recycle_scores_from_gamma = [&](const double r_proj_norm) {
    if (!enh.recycle_adaptive_keep_drop || recycle_k <= 0) {
      return;
    }
    const int k = recycle_k;
    if (static_cast<int>(recycle_score.size()) < k || static_cast<int>(recycle_drop_streak.size()) < k) {
      return;
    }

    double gamma_sq = 0.0;
    for (int j = 0; j < k; ++j) {
      const double g = recycle_y(j);
      gamma_sq += g * g;
    }

    const double r_true_norm = std::hypot(r_proj_norm, std::sqrt(gamma_sq));
    const double inv_r = (r_true_norm > 0.0) ? (1.0 / r_true_norm) : 0.0;

    double max_score = 0.0;
    const double a = enh.recycle_score_alpha;
    for (int j = 0; j < k; ++j) {
      const double rel = std::abs(recycle_y(j)) * inv_r;
      const size_t jj = static_cast<size_t>(j);
      // Use a leaky-max score: score <- max(a*score, rel).
      recycle_score[jj] = std::max(a * recycle_score[jj], rel);
      max_score = std::max(max_score, recycle_score[jj]);
    }

    const double thr = std::max(enh.recycle_drop_tol, enh.recycle_drop_rel * max_score);
    for (int j = 0; j < k; ++j) {
      const size_t jj = static_cast<size_t>(j);
      if (recycle_score[jj] < thr) {
        recycle_drop_streak[jj] += 1;
      } else {
        recycle_drop_streak[jj] = 0;
      }
    }
  };

  auto apply_recycle_keep_drop = [&]() {
    if (!enh.recycle_adaptive_keep_drop || recycle_k <= 0) {
      return;
    }
    const int k = recycle_k;
    if (static_cast<int>(recycle_score.size()) < k || static_cast<int>(recycle_drop_streak.size()) < k) {
      return;
    }

    std::vector<char> keep(static_cast<size_t>(k), 0);
    int kept = 0;
    for (int j = 0; j < k; ++j) {
      const size_t jj = static_cast<size_t>(j);
      if (recycle_drop_streak[jj] < enh.recycle_drop_window) {
        keep[jj] = 1;
        kept++;
      }
    }

    const int min_k = std::min(k, enh.recycle_drop_min_k);
    if (kept < min_k && min_k > 0) {
      std::vector<double> tmp(static_cast<size_t>(k), 0.0);
      for (int j = 0; j < k; ++j) {
        tmp[static_cast<size_t>(j)] = recycle_score[static_cast<size_t>(j)];
      }
      std::nth_element(tmp.begin(), tmp.begin() + (min_k - 1), tmp.end(),
                       [](const double x, const double y) { return x > y; });
      const double cutoff = tmp[static_cast<size_t>(min_k - 1)];
      kept = 0;
      for (int j = 0; j < k; ++j) {
        const size_t jj = static_cast<size_t>(j);
        keep[jj] = (recycle_score[jj] >= cutoff) ? 1 : 0;
        if (keep[jj]) {
          kept++;
        }
      }
    }

    if (kept >= k) {
      return;
    }

    int w = 0;
    for (int j = 0; j < k; ++j) {
      if (!keep[static_cast<size_t>(j)]) {
        continue;
      }
      if (w != j) {
        recycle_U.rslice(w) = recycle_U.rslice(j);
        recycle_C.rslice(w) = recycle_C.rslice(j);
        recycle_score[static_cast<size_t>(w)] = recycle_score[static_cast<size_t>(j)];
        recycle_drop_streak[static_cast<size_t>(w)] = recycle_drop_streak[static_cast<size_t>(j)];
      }
      w++;
    }

    recycle_k = w;
    ls.ws.recycle_k = recycle_k;
    if (enh.verbose && lhs.commu.task == 0) {
      std::fprintf(stderr, "[gmres_v] recycle keep/drop: k=%d win=%d\n", recycle_k, enh.recycle_drop_window);
    }
  };

  // Stagnation detection: terminate early if residual stops improving.
  // With restarted GMRES and a weak preconditioner, stagnation wastes
  // thousands of iterations (e.g. 51000 on an ill-conditioned system).
  double prev_fNorm = ls.fNorm;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;
  const bool allow_heuristic_early_stop = !ls.exact_convergence;

  // Adaptive early restart: within each restart cycle, the cost of
  // iteration i is O(i) for the GS orthogonalization (fused dot products
  // + vector updates against the entire Krylov basis).  If convergence
  // stalls at iteration k, continuing to sD wastes O((sD²-k²)/2) work.
  // Detect intra-restart stagnation: if the residual hasn't improved by
  // at least (1 - inner_stag_ratio) for inner_stag_window consecutive
  // iterations, break and restart.  Use env var to control.
  constexpr int inner_stag_window = 15;    // consecutive slow-progress iters
  constexpr double inner_stag_ratio = 0.9999; // require 0.01% improvement per iter
  // Only activate adaptive restart for restart dimensions larger than this
  // (small sD doesn't benefit and can lose convergence).
  constexpr int adaptive_min_sD = 50;
  const bool use_adaptive_restart = allow_heuristic_early_stop && (ls.sD >= adaptive_min_sD) &&
      !std::getenv("SVMP_FSILS_GMRES_NO_ADAPTIVE");

  for (int l = 0; l < ls.mItr; l++) {
    restart_cycles++;
    ls.dB = ls.fNorm;
    ls.itr++;
    if (l > 0) tp_restarts++;

    auto u_slice = u.rslice(0);
    apply_recycle_keep_drop();

    tp0 = TP();
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, X, u_slice);
    tp_spmv += TP() - tp0;

    tp0 = TP();
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, X, u_slice);
    tp_bc_mul += TP() - tp0;

    tp0 = TP();
    omp_la::omp_axpby_v(dof, nNo, u_slice, R, -1.0, u_slice);
    tp_vecops += TP() - tp0;

    if (has_coupled_bc) {
      tp0 = TP();
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, u_slice, u_slice);
      tp_bc_mul += TP() - tp0;
    }

    // Deflated/recycling GMRES: project residual and apply correction.
    if (recycle_k > 0) {
      tp0 = TP();
      fused_dot_v(dof, mynNo, recycle_C, recycle_k - 1, u_slice, h_col,
                  dot_thread.data(), thread_stride, num_threads);
      tp_dot_gs += TP() - tp0;

      tp0 = TP();
      bcast::fsils_bcast_v(recycle_k, h_col, lhs.commu);
      tp_allreduce += TP() - tp0;

      for (int j = 0; j < recycle_k; ++j) {
        recycle_y(j) = h_col[j];
      }

      tp0 = TP();
      fused_recon_v(dof, nNo, recycle_U, recycle_k - 1, X, recycle_y);
      tp_recon += TP() - tp0;

      tp0 = TP();
      fused_update_v_inplace(dof, nNo, recycle_C, recycle_k - 1, u_slice, h_col);
      tp_gs_update += TP() - tp0;
    }

    tp0 = TP();
    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u.rslice(0));
    tp_norm += TP() - tp0;

    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true; break;
    }

    update_recycle_scores_from_gamma(err[0]);

    tp0 = TP();
    omp_la::omp_mul_v(dof, nNo, 1.0 / err[0], u_slice);
    tp_vecops += TP() - tp0;

    // Adaptive early restart: track intra-restart convergence
    int inner_stag_count = 0;
    double prev_inner_err = std::abs(err[0]);

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;
      auto u_slice_prev = u.rslice(i);
      auto u_slice_next = u.rslice(i+1);

      tp0 = TP();
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, u_slice_prev, u_slice_next);
      tp_spmv += TP() - tp0;

      tp0 = TP();
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, u_slice_prev, u_slice_next);
      tp_bc_mul += TP() - tp0;

      if (has_coupled_bc) {
        tp0 = TP();
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, u_slice_next, u_slice_next);
        tp_bc_mul += TP() - tp0;
      }

      // Deflated/recycling GMRES: project the new Arnoldi vector.
      if (recycle_k > 0) {
        tp0 = TP();
        fused_dot_v(dof, mynNo, recycle_C, recycle_k - 1, u_slice_next, h_col,
                    dot_thread.data(), thread_stride, num_threads);
        tp_dot_gs += TP() - tp0;

        tp0 = TP();
        bcast::fsils_bcast_v(recycle_k, h_col, lhs.commu);
        tp_allreduce += TP() - tp0;

        // Store coupling coefficients B(:,i) = C^T * (A*v_i) before projection.
        // These are required to apply the corresponding U-space correction
        // so that the final residual minimizes the *true* residual norm
        // (not just the projected residual).
        for (int j = 0; j < recycle_k; ++j) {
          recycle_B[static_cast<size_t>(j) + static_cast<size_t>(recycle_k_req) * static_cast<size_t>(i)] = h_col[j];
        }

        tp0 = TP();
        fused_update_v_inplace(dof, nNo, recycle_C, recycle_k - 1, u_slice_next, h_col);
        tp_gs_update += TP() - tp0;
      }

      // --- Cache-blocked CGS Step 1 with Pythagorean trick ---
      // Fused dot: computes h_col[j] = dot(u[j], w) for j=0..i AND
      // h_col[i+1] = dot(w,w) in a single L1-tiled pass over the Krylov
      // basis.  Each tile loads w into a local buffer and streams u[0..i]
      // through L1, avoiding the per-vector DRAM round-trip of sequential
      // dot products.
      tp0 = TP();
      h_col[static_cast<size_t>(i+1)] = fused_dot_zz_v(dof, mynNo, u, i, u_slice_next, h_col,
                                                         dot_thread.data(), thread_stride, num_threads);
      tp_dot_gs += TP() - tp0;

      tp0 = TP();
      bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);
      tp_allreduce += TP() - tp0;

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      const double old_norm_sq = h_col[static_cast<size_t>(i+1)];

      // Cache-blocked axpy: w -= sum_j h_col[j] * u[j] in a single L1-tiled
      // pass.  Each tile loads w into a local buffer, accumulates all
      // subtractions from u[0..i], then writes back.
      // Note: fused_update_v_inplace does vbuf[k] -= hj * u[j][k], so
      // we pass h_col directly (positive values = subtract).
      tp0 = TP();
      const double first_norm = fused_update_norm_v(dof, nNo, mynNo, lhs.commu, u, i, u_slice_next, h_col);
      tp_gs_update += TP() - tp0;
      tp_norm += TP() - tp0;
      double new_norm = first_norm;
      const double new_norm_sq = new_norm * new_norm;

      // CGS2: Selective reorthogonalization (Pythagorean check)
      // tt_sq < proj_sq_norm means ||v_new|| < (1/sqrt(2))||v|| — loss of orthogonality.
      bool do_reorth = false;
      if (!ls.disable_reorth &&
          enh.reorth_mode != GmresEnhancements::ReorthMode::off &&
          std::isfinite(new_norm_sq)) {
        if (enh.use_mgs_dgks) {
          const double ratio2 = enh.mgs_dgks_ratio * enh.mgs_dgks_ratio;
          do_reorth = new_norm_sq < ratio2 * old_norm_sq;
        } else {
          do_reorth = new_norm_sq < proj_sq_norm;
        }
      }

      if (do_reorth) {
        tp_reorth_count++;

        tp0 = TP();
        h_col[static_cast<size_t>(i+1)] = fused_dot_zz_v(dof, mynNo, u, i, u_slice_next, h_col,
                                                           dot_thread.data(), thread_stride, num_threads);
        tp_dot_gs += TP() - tp0;

        tp0 = TP();
        bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);
        tp_allreduce += TP() - tp0;

        for (int j = 0; j <= i; ++j) {
          h(j,i) += h_col[j];
        }

        tp0 = TP();
        new_norm = fused_update_norm_v(dof, nNo, mynNo, lhs.commu, u, i, u_slice_next, h_col);
        tp_gs_update += TP() - tp0;
        tp_norm += TP() - tp0;
      }
      h(i+1,i) = new_norm;

      // Happy Breakdown Protection
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (h(i+1,i) > breakdown_tol) {
        tp0 = TP();
        omp_la::omp_mul_v(dof, nNo, 1.0/h(i+1,i), u_slice_next);
        tp_vecops += TP() - tp0;
      } else {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      tp0 = TP();
      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0; s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot; s(i) = h(i+1,i) / tmp_hypot;
      }

      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;

      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);
      tp_givens += TP() - tp0;

      if (std::abs(err(i+1)) < eps || breakdown) {
        ls.suc = true;
        break;
      }

      // Adaptive early restart: if convergence per iteration is very slow,
      // restart early.  The cost of iteration i is O(i) for GS operations,
      // so late iterations in a restart cycle are expensive.  Restarting
      // resets the O(i) cost back to O(0) while retaining the current
      // solution approximation.
      if (use_adaptive_restart && i >= inner_stag_window) {
        const double cur_err = std::abs(err(i+1));
        if (cur_err >= inner_stag_ratio * prev_inner_err) {
          inner_stag_count++;
        } else {
          inner_stag_count = 0;
        }
        prev_inner_err = cur_err;
        if (inner_stag_count >= inner_stag_window) {
          break;  // early restart
        }
      }
    }

    if (last_i >= ls.sD) last_i = ls.sD - 1;

    tp0 = TP();
    for (int i = 0; i <= last_i; i++) y(i) = err(i);

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) y(j) -= h(j,k)*y(k);
      y(j) /= h(j,j);
    }

    fused_recon_v(dof, nNo, u, last_i, X, y);
    tp_recon += TP() - tp0;

    // Recycling correction: X += U * ( -B * y ), where B(:,i) = C^T*(A*v_i).
    // This cancels the C-component of A*(V*y) and ensures ||err|| matches
    // the true residual norm for the augmented solution.
    if (recycle_k > 0) {
      for (int j = 0; j < recycle_k; ++j) {
        double sum = 0.0;
        for (int i = 0; i <= last_i; ++i) {
          const double yi = y(i);
          if (yi == 0.0) {
            continue;
          }
          sum += recycle_B[static_cast<size_t>(j) + static_cast<size_t>(recycle_k_req) * static_cast<size_t>(i)] * yi;
        }
        recycle_y(j) = -sum;
      }

      tp0 = TP();
      fused_recon_v(dof, nNo, recycle_U, recycle_k - 1, X, recycle_y);
      tp_recon += TP() - tp0;
    }

    ls.fNorm = std::abs(err(last_i+1));

    // Update recycling space from the just-built Krylov basis.
    if (recycle_k_req > 0 && enh.recycle_update_mode != GmresEnhancements::RecycleUpdateMode::off) {
      const bool do_update =
          (enh.recycle_update_mode == GmresEnhancements::RecycleUpdateMode::each_restart) ||
          (enh.recycle_update_mode == GmresEnhancements::RecycleUpdateMode::once_per_solve && !recycle_updated_this_solve);
      if (do_update) {
        update_recycle_space_from_arnoldi(last_i + 1);
        recycle_updated_this_solve = true;
      }
    }

    if (ls.suc) break;

    // Stagnation detection: if residual hasn't improved by at least
    // (1 - stagnation_ratio) for max_stagnation_restarts consecutive
    // restarts, the preconditioner is too weak for this system.
    if (allow_heuristic_early_stop && prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (allow_heuristic_early_stop && stagnation_count >= max_stagnation_restarts) {
      break;
    }
  }

  R = X;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB  = 10.0 * std::log(ls.fNorm / ls.dB);
  ls.stats.record_call(ls.itr - itr_before,
                       restart_cycles,
                       fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);

  // ===== PRINT TIMING PROFILE =====
  double tp_total = tp_spmv + tp_bc_mul + tp_dot_gs + tp_allreduce +
                    tp_gs_update + tp_norm + tp_vecops + tp_givens + tp_recon;
  if (enh.verbose && lhs.commu.task == 0 && tp_total > 0.0) {
    auto pct = [&](double t) { return 100.0 * t / tp_total; };
    fprintf(stderr,
      "\n=== GMRES_V TIMING PROFILE (rank 0) ===\n"
      "  Total GMRES time:     %10.6f s  (iters=%d, restarts=%d, reorth=%d)\n"
      "  SpMV:                 %10.6f s  (%5.1f%%)\n"
      "  BC multiply:          %10.6f s  (%5.1f%%)\n"
      "  GS dot products:      %10.6f s  (%5.1f%%)\n"
      "  MPI AllReduce:        %10.6f s  (%5.1f%%)\n"
      "  GS vector update:     %10.6f s  (%5.1f%%)\n"
      "  Norm computation:     %10.6f s  (%5.1f%%)\n"
      "  Vector ops (scale):   %10.6f s  (%5.1f%%)\n"
      "  Givens rotation:      %10.6f s  (%5.1f%%)\n"
      "  Reconstruction:       %10.6f s  (%5.1f%%)\n"
      "  dof=%d  nNo=%lld  mynNo=%lld  nnz=%lld  sD=%d\n"
      "========================================\n",
      tp_total, ls.itr, tp_restarts, tp_reorth_count,
      tp_spmv, pct(tp_spmv),
      tp_bc_mul, pct(tp_bc_mul),
      tp_dot_gs, pct(tp_dot_gs),
      tp_allreduce, pct(tp_allreduce),
      tp_gs_update, pct(tp_gs_update),
      tp_norm, pct(tp_norm),
      tp_vecops, pct(tp_vecops),
      tp_givens, pct(tp_givens),
      tp_recon, pct(tp_recon),
      dof, (long long)nNo, (long long)mynNo, (long long)lhs.nnz, ls.sD);
  }
  // ==================================
}

}; // namespace gmres
