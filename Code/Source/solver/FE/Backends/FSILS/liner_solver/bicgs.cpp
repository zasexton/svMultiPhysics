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

#include "bicgs.h"

#include "fsils_api.hpp"

#include "add_bc_mul.h"
#include "bcast.h"
#include "dot.h"
#include "gmres.h"
#include "norm.h"
#include "omp_la.h"
#include "spar_mul.h"

#include "Array3.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

namespace bicgs {

/// @brief Biconjugate-gradient stabilized algorithm for vector systems.
void bicgsv(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Array<double>& K, Array<double>& R)
{
  #define n_debug_bicgsv
  #ifdef debug_bicgsv
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  using namespace fe_fsi_linear_solver;

  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  #ifdef debug_bicgsv
  dmsg << "ls.mItr: " << ls.mItr;
  dmsg << "dof: " << dof;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  #endif

  ls.ws.ensure_bicgs_v(dof, nNo);
  auto& P = ls.ws.bicgs_P;
  auto& Rh = ls.ws.bicgs_Rh;
  auto& X = ls.ws.bicgs_X;
  auto& V = ls.ws.bicgs_V;
  auto& S = ls.ws.bicgs_S;
  auto& T = ls.ws.bicgs_T;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;
  ls.suc = false;
  double err = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  double errO = err;
  ls.iNorm = err;
  double eps = std::max(ls.absTol,ls.relTol*err);
  double rho = err*err;
  double beta = rho;
  X = 0.0;
  P = R;
  Rh = R;
  int i_itr = 1;
  #ifdef debug_bicgsv
  dmsg;
  dmsg << "err: " << err;
  dmsg << "eps: " << eps;
  #endif

  for (int i = 0; i < ls.mItr; i++) {
    #ifdef debug_bicgsv
    dmsg;
    dmsg << "----- i " << i+1 << " -----";
    dmsg << "err: " << err;
    dmsg << "eps: " << eps;
    #endif
    if (err < eps) {
      ls.suc = true;
      break;
    }

    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, K, P, V);
    double denom_alpha = dot::fsils_dot_v(dof, mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_v(dof, nNo, S, R, -alpha, V);

    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, K, S, T);
    double s_sq = dot::fsils_dot_v(dof, mynNo, lhs.commu, S, S);
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_v(dof, nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    double t_sq = dot::fsils_dot_v(dof, mynNo, lhs.commu, T, T);
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_v(dof, nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    double omega = dot::fsils_dot_v(dof, mynNo, lhs.commu, T, S) / t_sq;

    omp_la::omp_axpbypgz_v(dof, nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_v(dof, nNo, R, S, -omega, T);

    errO = err;
    err =  norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
    double rhoO  = rho;
    rho = dot::fsils_dot_v(dof, mynNo, lhs.commu, R, Rh);
    double denom_beta = rhoO * omega;
    if (std::abs(denom_beta) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    beta = rho*alpha / denom_beta;

    #ifdef debug_bicgsv
    dmsg << "alpha: " << alpha;
    dmsg << "omega: " << omega;
    dmsg << "rho: " << rho;
    dmsg << "beta: " << beta;
    #endif

    // P = R + beta*(P - omega*V)
    omp_la::omp_sum_v(dof, nNo, -omega, P, V);
    omp_la::omp_axpby_v(dof, nNo, P, R, beta, P);
    i_itr += 1;
  }

  R = X;
  ls.itr = i_itr - 1;
  ls.fNorm = err;
  ls.callD =  fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  #ifdef debug_bicgsv
  dmsg << "ls.itr: " << ls.itr;
  #endif

  if (errO < std::numeric_limits<double>::epsilon()) {
     ls.dB = 0.0;
  } else {
     ls.dB = 10.0 * std::log(err / errO);
  }
  ls.stats.record_call(ls.itr - itr_before,
                       /*restart_cycles=*/1,
                       fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);

}

//--------
// bicgss
//--------
//
void bicgss(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const Vector<double>& K, Vector<double>& R)
{
  #define n_debug_bicgss
  #ifdef debug_bicgss
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  using namespace fe_fsi_linear_solver;

  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  #ifdef debug_bicgss
  dmsg << "ls.mItr: " << ls.mItr;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  #endif

  ls.ws.ensure_bicgs_s(nNo);
  auto& P = ls.ws.bicgs_Ps;
  auto& Rh = ls.ws.bicgs_Rhs;
  auto& X = ls.ws.bicgs_Xs;
  auto& V = ls.ws.bicgs_Vs;
  auto& S = ls.ws.bicgs_Ss;
  auto& T = ls.ws.bicgs_Ts;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;
  ls.suc = false;
  double err = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  double errO = err;
  ls.iNorm = err;
  double eps = std::max(ls.absTol,ls.relTol*err);
  double rho = err*err;
  double beta = rho;
  X = 0.0;
  P = R;
  Rh = R;
  int i_itr = 1;
  #ifdef debug_bicgss
  dmsg;
  dmsg << "err: " << err;
  dmsg << "eps: " << eps;
  #endif

  for (int i = 0; i < ls.mItr; i++) {
    #ifdef debug_bicgss
    dmsg;
    dmsg << "----- i " << i+1 << " -----";
    dmsg << "err: " << err;
    dmsg << "eps: " << eps;
    #endif
    if (err < eps) {
      ls.suc = true;
      break;
    }

    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, K, P, V);
    double denom_alpha = dot::fsils_dot_s(mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_s(nNo, S, R, -alpha, V);

    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, K, S, T);
    double s_sq = dot::fsils_dot_s(mynNo, lhs.commu, S, S);
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_s(nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    double t_sq = dot::fsils_dot_s(mynNo, lhs.commu, T, T);
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    double omega = dot::fsils_dot_s(mynNo, lhs.commu, T, S) / t_sq;

    omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_s(nNo, R, S, -omega, T);

    errO = err;
    err =  norm::fsi_ls_norms(mynNo, lhs.commu, R);
    double rhoO  = rho;
    rho = dot::fsils_dot_s(mynNo, lhs.commu, R, Rh);
    double denom_beta = rhoO * omega;
    if (std::abs(denom_beta) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    beta = rho*alpha / denom_beta;

    #ifdef debug_bicgss
    dmsg << "alpha: " << alpha;
    dmsg << "omega: " << omega;
    dmsg << "rho: " << rho;
    dmsg << "beta: " << beta;
    #endif

    // P = R + beta*(P - omega*V)
    omp_la::omp_sum_s(nNo, -omega, P, V);
    omp_la::omp_axpby_s(nNo, P, R, beta, P);
    i_itr += 1;
  }

  R = X;
  ls.itr = i_itr - 1;
  ls.fNorm = err;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  #ifdef debug_bicgss
  dmsg << "ls.itr: " << ls.itr;
  #endif

  if (errO < std::numeric_limits<double>::epsilon()) {
     ls.dB = 0.0;
  } else {
     ls.dB = 10.0 * std::log(err / errO);
  }
  ls.stats.record_call(ls.itr - itr_before,
                       /*restart_cycles=*/1,
                       fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);
}

namespace {

using fe_fsi_linear_solver::fsils_int;

struct SchurPreconditionerData {
  Array<double> point_inv;
  Array<double> ilu_factors;
  Array<double> ilu_diag_inv;
  bool use_ilu{false};
};

[[nodiscard]] inline const double* block_ptr(const Array<double>& A, int block_entries, fsils_int index)
{
  return A.data() + static_cast<size_t>(block_entries) * static_cast<size_t>(index);
}

[[nodiscard]] inline double* block_ptr(Array<double>& A, int block_entries, fsils_int index)
{
  return A.data() + static_cast<size_t>(block_entries) * static_cast<size_t>(index);
}

void set_zero(double* data, int count)
{
  std::fill(data, data + count, 0.0);
}

void set_identity(int n, double* data)
{
  set_zero(data, n * n);
  for (int i = 0; i < n; ++i) {
    data[i * n + i] = 1.0;
  }
}

[[nodiscard]] double safe_inverse(double value)
{
  return (std::abs(value) > 1e-12) ? 1.0 / value : 1.0;
}

bool invert_dense_block(int n, const double* A, double* inv)
{
  std::vector<double> work(static_cast<size_t>(n) * static_cast<size_t>(n));
  std::copy(A, A + n * n, work.begin());
  set_identity(n, inv);

  for (int col = 0; col < n; ++col) {
    int pivot_row = col;
    double pivot_abs = std::abs(work[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
    for (int row = col + 1; row < n; ++row) {
      const double candidate = std::abs(work[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
      if (candidate > pivot_abs) {
        pivot_abs = candidate;
        pivot_row = row;
      }
    }

    if (pivot_abs < 1e-14) {
      return false;
    }

    if (pivot_row != col) {
      for (int j = 0; j < n; ++j) {
        std::swap(work[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)],
                  work[static_cast<size_t>(pivot_row) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
        std::swap(inv[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)],
                  inv[static_cast<size_t>(pivot_row) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
      }
    }

    const double pivot = work[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(col)];
    for (int j = 0; j < n; ++j) {
      work[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)] /= pivot;
      inv[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)] /= pivot;
    }

    for (int row = 0; row < n; ++row) {
      if (row == col) {
        continue;
      }
      const double factor = work[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)];
      if (std::abs(factor) <= 0.0) {
        continue;
      }
      for (int j = 0; j < n; ++j) {
        work[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(j)] -=
            factor * work[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)];
        inv[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(j)] -=
            factor * inv[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)];
      }
    }
  }

  return true;
}

void multiply_blocks(const double* A, int rows, int inner,
                     const double* B, int cols,
                     double* C)
{
  set_zero(C, rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k < inner; ++k) {
      const double a = A[i * inner + k];
      for (int j = 0; j < cols; ++j) {
        C[i * cols + j] += a * B[k * cols + j];
      }
    }
  }
}

void multiply_block_vector(const double* A, int rows, int cols,
                           const double* x,
                           double* y)
{
  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += A[i * cols + j] * x[j];
    }
    y[i] = sum;
  }
}

void subtract_block_vector_product(double* y,
                                   const double* A, int rows, int cols,
                                   const double* x)
{
  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += A[i * cols + j] * x[j];
    }
    y[i] -= sum;
  }
}

[[nodiscard]] fsils_int find_col_in_row(const Array<fsils_int>& rowPtr,
                                        const Vector<fsils_int>& colPtr,
                                        fsils_int row,
                                        fsils_int col)
{
  const fsils_int start = rowPtr(0, row);
  const fsils_int end = rowPtr(1, row);
  const fsils_int* begin = colPtr.data() + start;
  const fsils_int* finish = colPtr.data() + end + 1;
  const auto it = std::lower_bound(begin, finish, col);
  if (it == finish || *it != col) {
    return -1;
  }
  return static_cast<fsils_int>(it - colPtr.data());
}

void build_point_inverse_blocks(const Vector<fsils_int>& diagPtr,
                                int block_size,
                                fsils_int nNo,
                                const Array<double>& values,
                                bool diagonal_only,
                                Array<double>& inv_blocks)
{
  const int block_entries = block_size * block_size;
  inv_blocks.resize(block_entries, nNo);

  std::vector<double> diag_block(static_cast<size_t>(block_entries), 0.0);
  std::vector<double> inv_block(static_cast<size_t>(block_entries), 0.0);
  for (fsils_int i = 0; i < nNo; ++i) {
    const fsils_int diag_nz = diagPtr(i);
    double* out = block_ptr(inv_blocks, block_entries, i);
    if (diagonal_only) {
      set_zero(out, block_entries);
      for (int d = 0; d < block_size; ++d) {
        out[d * block_size + d] = safe_inverse(values(d * block_size + d, diag_nz));
      }
      continue;
    }

    const double* src = block_ptr(values, block_entries, diag_nz);
    std::copy(src, src + block_entries, diag_block.begin());
    if (!invert_dense_block(block_size, diag_block.data(), inv_block.data())) {
      set_zero(out, block_entries);
      for (int d = 0; d < block_size; ++d) {
        out[d * block_size + d] = safe_inverse(diag_block[static_cast<size_t>(d) * static_cast<size_t>(block_size) +
                                                          static_cast<size_t>(d)]);
      }
    } else {
      std::copy(inv_block.begin(), inv_block.end(), out);
    }
  }
}

void factorize_block_ilu0(const Array<fsils_int>& rowPtr,
                          const Vector<fsils_int>& colPtr,
                          const Vector<fsils_int>& diagPtr,
                          fsils_int nNo,
                          int block_size,
                          const Array<double>& values,
                          Array<double>& factors,
                          Array<double>& diag_inv)
{
  const int block_entries = block_size * block_size;
  factors = values;
  diag_inv.resize(block_entries, nNo);

  std::vector<double> lij(static_cast<size_t>(block_entries), 0.0);
  std::vector<double> prod(static_cast<size_t>(block_entries), 0.0);
  std::vector<double> fallback_diag(static_cast<size_t>(block_entries), 0.0);
  std::vector<double> inv_block(static_cast<size_t>(block_entries), 0.0);

  for (fsils_int row = 0; row < nNo; ++row) {
    const fsils_int diag_nz = diagPtr(row);
    for (fsils_int p = rowPtr(0, row); p < diag_nz; ++p) {
      const fsils_int col = colPtr(p);
      multiply_blocks(block_ptr(factors, block_entries, p), block_size, block_size,
                      block_ptr(diag_inv, block_entries, col), block_size,
                      lij.data());
      std::copy(lij.begin(), lij.end(), block_ptr(factors, block_entries, p));

      for (fsils_int q = diagPtr(col) + 1; q <= rowPtr(1, col); ++q) {
        const fsils_int target_col = colPtr(q);
        const fsils_int target = find_col_in_row(rowPtr, colPtr, row, target_col);
        if (target < 0) {
          continue;
        }
        multiply_blocks(lij.data(), block_size, block_size,
                        block_ptr(factors, block_entries, q), block_size,
                        prod.data());
        double* target_block = block_ptr(factors, block_entries, target);
        for (int e = 0; e < block_entries; ++e) {
          target_block[e] -= prod[static_cast<size_t>(e)];
        }
      }
    }

    double* diag_out = block_ptr(diag_inv, block_entries, row);
    const double* diag_block = block_ptr(factors, block_entries, diag_nz);
    if (!invert_dense_block(block_size, diag_block, diag_out)) {
      const double* fallback = block_ptr(values, block_entries, diag_nz);
      std::copy(fallback, fallback + block_entries, fallback_diag.begin());
      if (!invert_dense_block(block_size, fallback_diag.data(), inv_block.data())) {
        set_zero(diag_out, block_entries);
        for (int d = 0; d < block_size; ++d) {
          diag_out[d * block_size + d] =
              safe_inverse(fallback_diag[static_cast<size_t>(d) * static_cast<size_t>(block_size) +
                                         static_cast<size_t>(d)]);
        }
      } else {
        std::copy(inv_block.begin(), inv_block.end(), diag_out);
      }
    }
  }
}

void apply_point_block_inverse(const Array<double>& inv_blocks,
                               int block_size,
                               fsils_int nNo,
                               Array<double>& x)
{
  std::vector<double> tmp(static_cast<size_t>(block_size), 0.0);
  for (fsils_int i = 0; i < nNo; ++i) {
    multiply_block_vector(block_ptr(inv_blocks, block_size * block_size, i), block_size, block_size,
                          x.data() + static_cast<size_t>(i) * static_cast<size_t>(block_size),
                          tmp.data());
    for (int k = 0; k < block_size; ++k) {
      x(k, i) = tmp[static_cast<size_t>(k)];
    }
  }
}

void apply_block_ilu0(const Array<fsils_int>& rowPtr,
                      const Vector<fsils_int>& colPtr,
                      const Vector<fsils_int>& diagPtr,
                      int block_size,
                      fsils_int nNo,
                      const Array<double>& factors,
                      const Array<double>& diag_inv,
                      Array<double>& x)
{
  Array<double> y(block_size, nNo);
  y = x;

  std::vector<double> tmp(static_cast<size_t>(block_size), 0.0);
  for (fsils_int row = 0; row < nNo; ++row) {
    for (int k = 0; k < block_size; ++k) {
      tmp[static_cast<size_t>(k)] = y(k, row);
    }
    for (fsils_int p = rowPtr(0, row); p < diagPtr(row); ++p) {
      subtract_block_vector_product(tmp.data(),
                                    block_ptr(factors, block_size * block_size, p), block_size, block_size,
                                    y.data() + static_cast<size_t>(colPtr(p)) * static_cast<size_t>(block_size));
    }
    for (int k = 0; k < block_size; ++k) {
      y(k, row) = tmp[static_cast<size_t>(k)];
    }
  }

  for (fsils_int row = nNo; row-- > 0;) {
    for (int k = 0; k < block_size; ++k) {
      tmp[static_cast<size_t>(k)] = y(k, row);
    }
    for (fsils_int p = diagPtr(row) + 1; p <= rowPtr(1, row); ++p) {
      subtract_block_vector_product(tmp.data(),
                                    block_ptr(factors, block_size * block_size, p), block_size, block_size,
                                    x.data() + static_cast<size_t>(colPtr(p)) * static_cast<size_t>(block_size));
    }
    multiply_block_vector(block_ptr(diag_inv, block_size * block_size, row), block_size, block_size,
                          tmp.data(),
                          x.data() + static_cast<size_t>(row) * static_cast<size_t>(block_size));
  }
}

void build_momentum_hat_blocks(const Array<fsils_int>& rowPtr,
                               const Vector<fsils_int>& colPtr,
                               const Vector<fsils_int>& diagPtr,
                               fsils_int nNo,
                               int mom_ncomp,
                               const Array<double>& K,
                               fe_fsi_linear_solver::SchurMomentumApproximationType approx,
                               Array<double>& momentum_hat_blocks)
{
  using fe_fsi_linear_solver::SchurMomentumApproximationType;

  switch (approx) {
    case SchurMomentumApproximationType::DIAG_K:
      build_point_inverse_blocks(diagPtr, mom_ncomp, nNo, K, /*diagonal_only=*/true, momentum_hat_blocks);
      return;
    case SchurMomentumApproximationType::BLOCKDIAG_K:
      build_point_inverse_blocks(diagPtr, mom_ncomp, nNo, K, /*diagonal_only=*/false, momentum_hat_blocks);
      return;
    case SchurMomentumApproximationType::ILU_K:
    case SchurMomentumApproximationType::ASM_K: {
      // Zero-overlap ASM currently shares the same rank-local ILU(0) block extraction path.
      Array<double> factors;
      factorize_block_ilu0(rowPtr, colPtr, diagPtr, nNo, mom_ncomp, K, factors, momentum_hat_blocks);
      return;
    }
  }
}

void assemble_algebraic_schur(const Array<fsils_int>& rowPtr,
                              const Vector<fsils_int>& colPtr,
                              fsils_int nNo,
                              int mom_ncomp,
                              int con_ncomp,
                              const Array<double>& D,
                              const Array<double>& G,
                              const Array<double>& L,
                              const Array<double>& momentum_hat_blocks,
                              Array<double>& shat)
{
  const int con_block_entries = con_ncomp * con_ncomp;
  const int mixed_block_entries = con_ncomp * mom_ncomp;
  shat = L;

  std::vector<double> dh(static_cast<size_t>(mixed_block_entries), 0.0);
  std::vector<double> contrib(static_cast<size_t>(con_block_entries), 0.0);
  for (fsils_int row = 0; row < nNo; ++row) {
    for (fsils_int p = rowPtr(0, row); p <= rowPtr(1, row); ++p) {
      const fsils_int middle = colPtr(p);
      multiply_blocks(block_ptr(D, mixed_block_entries, p), con_ncomp, mom_ncomp,
                      block_ptr(momentum_hat_blocks, mom_ncomp * mom_ncomp, middle), mom_ncomp,
                      dh.data());

      for (fsils_int q = rowPtr(0, middle); q <= rowPtr(1, middle); ++q) {
        const fsils_int col = colPtr(q);
        const fsils_int target = find_col_in_row(rowPtr, colPtr, row, col);
        if (target < 0) {
          continue;
        }
        multiply_blocks(dh.data(), con_ncomp, mom_ncomp,
                        block_ptr(G, mom_ncomp * con_ncomp, q), con_ncomp,
                        contrib.data());
        double* target_block = block_ptr(shat, con_block_entries, target);
        for (int e = 0; e < con_block_entries; ++e) {
          target_block[e] -= contrib[static_cast<size_t>(e)];
        }
      }
    }
  }
}

SchurPreconditionerData build_schur_preconditioner(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                                   fe_fsi_linear_solver::FSILS_subLsType& ls,
                                                   int mom_ncomp,
                                                   int con_ncomp,
                                                   const Array<double>& K,
                                                   const Array<double>& D,
                                                   const Array<double>& G,
                                                   const Array<double>& L)
{
  using fe_fsi_linear_solver::SchurPreconditionerType;

  SchurPreconditionerData pc{};
  const auto preconditioner = ls.schur_preconditioner;
  const bool diagonal_only = (preconditioner == SchurPreconditionerType::DIAG_L);
  build_point_inverse_blocks(lhs.diagPtr, con_ncomp, lhs.nNo, L, diagonal_only, pc.point_inv);

  if (preconditioner == SchurPreconditionerType::ILU_L) {
    factorize_block_ilu0(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, con_ncomp, L,
                         pc.ilu_factors, pc.ilu_diag_inv);
    pc.use_ilu = true;
    return pc;
  }

  if (preconditioner == SchurPreconditionerType::ALGEBRAIC_SHAT) {
    Array<double> momentum_hat_blocks;
    build_momentum_hat_blocks(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, mom_ncomp, K,
                              ls.schur_momentum_approximation, momentum_hat_blocks);

    Array<double> shat(L.nrows(), L.ncols());
    assemble_algebraic_schur(lhs.rowPtr, lhs.colPtr, lhs.nNo, mom_ncomp, con_ncomp,
                             D, G, L, momentum_hat_blocks, shat);
    factorize_block_ilu0(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, con_ncomp, shat,
                         pc.ilu_factors, pc.ilu_diag_inv);
    pc.use_ilu = true;
  }

  return pc;
}

void apply_schur_preconditioner(const Array<fsils_int>& rowPtr,
                                const Vector<fsils_int>& colPtr,
                                const Vector<fsils_int>& diagPtr,
                                const SchurPreconditionerData& pc,
                                int con_ncomp,
                                fsils_int nNo,
                                Array<double>& x)
{
  if (pc.use_ilu) {
    apply_block_ilu0(rowPtr, colPtr, diagPtr, con_ncomp, nNo, pc.ilu_factors, pc.ilu_diag_inv, x);
    return;
  }
  apply_point_block_inverse(pc.point_inv, con_ncomp, nNo, x);
}

void copy_scalar_vector_to_array(const Vector<double>& src, Array<double>& dst)
{
  const fsils_int nNo = src.size();
  dst.resize(1, nNo);
  for (fsils_int i = 0; i < nNo; ++i) {
    dst(0, i) = src(i);
  }
}

void copy_scalar_array_to_vector(const Array<double>& src, Vector<double>& dst)
{
  const fsils_int nNo = src.ncols();
  if (dst.size() != nNo) {
    dst.resize(nNo);
  }
  for (fsils_int i = 0; i < nNo; ++i) {
    dst(i) = src(0, i);
  }
}

void copy_scalar_vector_to_matrix(const Vector<double>& src, Array<double>& dst)
{
  const fsils_int nnz = src.size();
  dst.resize(1, nnz);
  for (fsils_int i = 0; i < nnz; ++i) {
    dst(0, i) = src(i);
  }
}

void schur_impl(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                fe_fsi_linear_solver::FSILS_subLsType& ls,
                fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
                int mom_ncomp,
                int con_ncomp,
                const Array<double>& K,
                const Array<double>& D,
                const Array<double>& G,
                const Array<double>& L,
                Array<double>& R)
{
  using namespace fe_fsi_linear_solver;

  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_bicgs_v(con_ncomp, nNo);
  auto& P = ls.ws.bicgs_P;
  auto& Rh = ls.ws.bicgs_Rh;
  auto& X = ls.ws.bicgs_X;
  auto& V = ls.ws.bicgs_V;
  auto& S = ls.ws.bicgs_S;
  auto& T = ls.ws.bicgs_T;

  Array<double> GP(mom_ncomp, nNo);
  Array<double> HGP(mom_ncomp, nNo);
  Array<double> SP(con_ncomp, nNo);
  Array<double> DGP(con_ncomp, nNo);
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;
  const double setup_t0 = fe_fsi_linear_solver::fsils_cpu_t();
  const auto pc = build_schur_preconditioner(lhs, ls, mom_ncomp, con_ncomp, K, D, G, L);
  const double pc_setup_time = fe_fsi_linear_solver::fsils_cpu_t() - setup_t0;
  apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, pc, con_ncomp, nNo, R);

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;

  double err = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, R);
  double errO = err;
  ls.iNorm = err;
  const double eps = std::max(ls.absTol, ls.relTol * err);
  double rho = err * err;
  double beta = rho;

  X = 0.0;
  P = R;
  Rh = R;
  int i_itr = 1;

  auto apply_schur_operator = [&](const Array<double>& in_vec, Array<double>& out_vec) {
    spar_mul::fsils_spar_mul_rect_vv_fused(lhs, lhs.rowPtr, lhs.colPtr,
        mom_ncomp, con_ncomp, G, L, in_vec, GP, SP);

    HGP = GP;
    gmres::gmres_v(lhs, momentum_ls, mom_ncomp, K, HGP);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      // The nested momentum solve returns a distributed overlap vector.
      // Synchronize shared nodes before reusing it as the input to D*(K^-1*G).
      fsils_commuv(lhs, mom_ncomp, HGP);
    }

    spar_mul::fsils_spar_mul_rect(lhs, lhs.rowPtr, lhs.colPtr,
        con_ncomp, mom_ncomp, D, HGP, DGP);

    #pragma omp parallel for schedule(static)
    for (fsils_int i = 0; i < nNo; ++i) {
      for (int k = 0; k < con_ncomp; ++k) {
        out_vec(k, i) = SP(k, i) - DGP(k, i);
      }
    }

    apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, pc, con_ncomp, nNo, out_vec);
  };

  for (int i = 0; i < ls.mItr; ++i) {
    if (err < eps) {
      ls.suc = true;
      break;
    }

    apply_schur_operator(P, V);

    const double denom_alpha = dot::fsils_dot_v(con_ncomp, mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) <
        std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) {
      break;
    }
    const double alpha = rho / denom_alpha;

    omp_la::omp_axpby_v(con_ncomp, nNo, S, R, -alpha, V);
    apply_schur_operator(S, T);

    const double s_sq = dot::fsils_dot_v(con_ncomp, mynNo, lhs.commu, S, S);
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_v(con_ncomp, nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    const double t_sq = dot::fsils_dot_v(con_ncomp, mynNo, lhs.commu, T, T);
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_v(con_ncomp, nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    const double omega = dot::fsils_dot_v(con_ncomp, mynNo, lhs.commu, T, S) / t_sq;

    omp_la::omp_axpbypgz_v(con_ncomp, nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_v(con_ncomp, nNo, R, S, -omega, T);

    errO = err;
    err = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, R);

    const double rhoO = rho;
    rho = dot::fsils_dot_v(con_ncomp, mynNo, lhs.commu, R, Rh);

    const double denom_beta = rhoO * omega;
    if (std::abs(denom_beta) <
        std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) {
      break;
    }
    beta = (rho * alpha) / denom_beta;

    omp_la::omp_sum_v(con_ncomp, nNo, -omega, P, V);
    omp_la::omp_axpby_v(con_ncomp, nNo, P, R, beta, P);

    i_itr += 1;
  }

  R = X;
  ls.itr = i_itr - 1;
  ls.fNorm = err;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB = (errO < std::numeric_limits<double>::epsilon()) ? 0.0 : 10.0 * std::log(err / errO);
  ls.stats.record_call(ls.itr - itr_before,
                       /*restart_cycles=*/1,
                       fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       pc_setup_time,
                       ls.callD - callD_before);
}

} // namespace

//--------
// schur
//--------
/// @brief BiCGStab Schur complement solver for (L - D*H*G) P = R.
/// Handles asymmetric saddle-point systems where D != -G^T (e.g., stabilized
/// formulations). The Schur preconditioner is selected via FSILS BlockSchur
/// configuration and can be diagonal/block-diagonal/ILU on L or an algebraic
/// sparse Schur approximation built from K, D, G, and L.
void schur(fe_fsi_linear_solver::FSILS_lhsType& lhs,
           fe_fsi_linear_solver::FSILS_subLsType& ls,
           fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
           const int nsd,
           const Array<double>& K, const Array<double>& D, const Array<double>& G,
           const Vector<double>& L, Vector<double>& R)
{
  Array<double> L_block;
  Array<double> R_block;
  copy_scalar_vector_to_matrix(L, L_block);
  copy_scalar_vector_to_array(R, R_block);
  schur_impl(lhs, ls, momentum_ls, nsd, /*con_ncomp=*/1, K, D, G, L_block, R_block);
  copy_scalar_array_to_vector(R_block, R);
}

//--------
// schur_mc
//--------
/// @brief Multi-component BiCGStab Schur complement solver for (L - D*H*G) P = R.
/// D(con_ncomp*mom_ncomp, nnz), G(mom_ncomp*con_ncomp, nnz), L(con_ncomp*con_ncomp, nnz).
/// R(con_ncomp, nNo) is both RHS input and solution output.
void schur_mc(fe_fsi_linear_solver::FSILS_lhsType& lhs,
              fe_fsi_linear_solver::FSILS_subLsType& ls,
              fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
              int mom_ncomp, int con_ncomp,
              const Array<double>& K, const Array<double>& D, const Array<double>& G,
              const Array<double>& L, Array<double>& R)
{
  schur_impl(lhs, ls, momentum_ls, mom_ncomp, con_ncomp, K, D, G, L, R);
}

} // namespace bicgs
