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

#include <unordered_map>
#include "spar_mul.h"

#include "Array3.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <string>
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

struct SchurSparsityControl {
  double momentum_inverse_offdiag_score_min{0.0};
  int momentum_inverse_max_offdiag_per_row{-1};
  double shat_row_score_min{0.0};
  int shat_max_offdiag_per_row{-1};
  int guaranteed_offdiag_per_row{2};
  int operator_refinement_steps{1};
  double operator_refinement_omega{1.0};
};

[[nodiscard]] const SchurSparsityControl& schur_sparsity_control()
{
  static const SchurSparsityControl cfg = []() {
    SchurSparsityControl c{};
    c.momentum_inverse_offdiag_score_min =
        parse_double_env("SVMP_FSILS_BLOCKSCHUR_MOMINV_DROP_SCORE", 0.0);
    if (!(c.momentum_inverse_offdiag_score_min >= 0.0 && std::isfinite(c.momentum_inverse_offdiag_score_min))) {
      c.momentum_inverse_offdiag_score_min = 0.0;
    }

    c.momentum_inverse_max_offdiag_per_row =
        parse_int_env("SVMP_FSILS_BLOCKSCHUR_MOMINV_MAX_OFFDIAG", -1);
    if (c.momentum_inverse_max_offdiag_per_row < -1) {
      c.momentum_inverse_max_offdiag_per_row = -1;
    }

    c.shat_row_score_min =
        parse_double_env("SVMP_FSILS_BLOCKSCHUR_SHAT_ROW_DROP_SCORE", 0.0);
    if (!(c.shat_row_score_min >= 0.0 && std::isfinite(c.shat_row_score_min))) {
      c.shat_row_score_min = 0.0;
    }

    c.shat_max_offdiag_per_row =
        parse_int_env("SVMP_FSILS_BLOCKSCHUR_SHAT_MAX_OFFDIAG", -1);
    if (c.shat_max_offdiag_per_row < -1) {
      c.shat_max_offdiag_per_row = -1;
    }

    c.guaranteed_offdiag_per_row =
        std::max(0, parse_int_env("SVMP_FSILS_BLOCKSCHUR_SHAT_GUARANTEED_OFFDIAG", 2));
    c.operator_refinement_steps =
        std::max(0, parse_int_env("SVMP_FSILS_BLOCKSCHUR_SHAT_OPERATOR_REFINE_STEPS", 1));
    c.operator_refinement_omega =
        parse_double_env("SVMP_FSILS_BLOCKSCHUR_SHAT_OPERATOR_REFINE_OMEGA", 0.5);
    if (!(c.operator_refinement_omega > 0.0 && std::isfinite(c.operator_refinement_omega))) {
      c.operator_refinement_omega = 1.0;
    }
    return c;
  }();
  return cfg;
}

struct MomentumHatData {
  Array<double> point_inv;
  Array<double> ilu_factors;
  Array<double> ilu_diag_inv;
  Array<double> sparse_inverse;
  bool use_ilu{false};
};

struct SchurPreconditionerData {
  Array<double> point_inv;
  Array<double> ilu_factors;
  Array<double> ilu_diag_inv;
  bool use_ilu{false};
  MomentumHatData momentum_hat;
  const Array<double>* operator_D{nullptr};
  const Array<double>* operator_G{nullptr};
  Array<double> operator_L_storage;
  const Array<double>* operator_L{nullptr};
  int operator_mom_ncomp{0};
  int operator_con_ncomp{0};
  int operator_refinement_steps{0};
  double operator_refinement_omega{1.0};
  std::vector<Array<double>> low_rank_right;
  std::vector<Array<double>> low_rank_preconditioned_left;
  std::vector<double> low_rank_inner_inv;
  std::vector<Array<double>> coarse_basis;
  std::vector<Array<double>> coarse_operator_basis;
  std::vector<double> coarse_inner_inv;
  Array<double> scratch_rhs;
  Array<double> scratch_ax;
  Array<double> scratch_gp;
  Array<double> scratch_hgp;
  Array<double> scratch_sp;
  Array<double> scratch_dgp;
  Array<double> scratch_correction;
  Array<double> scratch_residual;
};

struct SchurSolveCacheEntry {
  bool valid{false};
  int mom_ncomp{0};
  int con_ncomp{0};
  fsils_int nNo{0};
  SchurPreconditionerData preconditioner;
  Array<double> previous_solution;
};

[[nodiscard]] std::unordered_map<const fe_fsi_linear_solver::FSILS_subLsType*, SchurSolveCacheEntry>& schur_cache_registry()
{
  static std::unordered_map<const fe_fsi_linear_solver::FSILS_subLsType*, SchurSolveCacheEntry> cache;
  return cache;
}

[[nodiscard]] SchurSolveCacheEntry& schur_cache_entry(fe_fsi_linear_solver::FSILS_subLsType& ls)
{
  return schur_cache_registry()[&ls];
}

[[nodiscard]] bool has_cached_schur_solution(const SchurSolveCacheEntry& cache, int con_ncomp, fsils_int nNo)
{
  return cache.previous_solution.nrows() == con_ncomp && cache.previous_solution.ncols() == nNo;
}

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

void multiply_block_vector_transpose(const double* A, int rows, int cols,
                                     const double* x,
                                     double* y)
{
  for (int j = 0; j < cols; ++j) {
    double sum = 0.0;
    for (int i = 0; i < rows; ++i) {
      sum += A[i * cols + j] * x[i];
    }
    y[j] = sum;
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

void subtract_transpose_block_vector_product(double* y,
                                             const double* A, int rows, int cols,
                                             const double* x)
{
  for (int j = 0; j < cols; ++j) {
    double sum = 0.0;
    for (int i = 0; i < rows; ++i) {
      sum += A[i * cols + j] * x[i];
    }
    y[j] -= sum;
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

void build_sparse_block_diagonal_inverse(const Vector<fsils_int>& diagPtr,
                                       int block_size,
                                       fsils_int nNo,
                                       fsils_int nnz,
                                       const Array<double>& diag_inv,
                                       Array<double>& sparse_inverse)
{
  const int block_entries = block_size * block_size;
  sparse_inverse.resize(block_entries, nnz);
  sparse_inverse = 0.0;
  for (fsils_int row = 0; row < nNo; ++row) {
    const fsils_int diag_nz = diagPtr(row);
    double* out = block_ptr(sparse_inverse, block_entries, diag_nz);
    const double* src = block_ptr(diag_inv, block_entries, row);
    std::copy(src, src + block_entries, out);
  }
}

void build_sparse_block_ilu_inverse(const Array<fsils_int>& rowPtr,
                                    const Vector<fsils_int>& colPtr,
                                    const Vector<fsils_int>& diagPtr,
                                    fsils_int nNo,
                                    fsils_int nnz,
                                    int block_size,
                                    const Array<double>& factors,
                                    const Array<double>& diag_inv,
                                    Array<double>& sparse_inverse)
{
  const int block_entries = block_size * block_size;
  sparse_inverse.resize(block_entries, nnz);
  sparse_inverse = 0.0;

  std::vector<double> tmp_left(static_cast<size_t>(block_entries), 0.0);
  std::vector<double> tmp_right(static_cast<size_t>(block_entries), 0.0);
  for (fsils_int row = 0; row < nNo; ++row) {
    const fsils_int diag_nz = diagPtr(row);
    const double* drow_inv = block_ptr(diag_inv, block_entries, row);
    std::copy(drow_inv, drow_inv + block_entries, block_ptr(sparse_inverse, block_entries, diag_nz));

    for (fsils_int p = rowPtr(0, row); p < diag_nz; ++p) {
      multiply_blocks(drow_inv, block_size, block_size,
                      block_ptr(factors, block_entries, p), block_size,
                      tmp_left.data());
      double* out = block_ptr(sparse_inverse, block_entries, p);
      for (int e = 0; e < block_entries; ++e) {
        out[e] = -tmp_left[static_cast<size_t>(e)];
      }
    }

    for (fsils_int p = diag_nz + 1; p <= rowPtr(1, row); ++p) {
      const fsils_int col = colPtr(p);
      multiply_blocks(drow_inv, block_size, block_size,
                      block_ptr(factors, block_entries, p), block_size,
                      tmp_left.data());
      multiply_blocks(tmp_left.data(), block_size, block_size,
                      block_ptr(diag_inv, block_entries, col), block_size,
                      tmp_right.data());
      double* out = block_ptr(sparse_inverse, block_entries, p);
      for (int e = 0; e < block_entries; ++e) {
        out[e] = -tmp_right[static_cast<size_t>(e)];
      }
    }
  }
}

[[nodiscard]] bool block_is_zero(const double* block, int count)
{
  for (int i = 0; i < count; ++i) {
    if (std::abs(block[i]) > 1e-30) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double block_max_abs(const double* block, int count)
{
  double value = 0.0;
  for (int i = 0; i < count; ++i) {
    value = std::max(value, std::abs(block[i]));
  }
  return value;
}

void prune_sparse_momentum_hat(const Array<fsils_int>& rowPtr,
                               const Vector<fsils_int>& colPtr,
                               const Vector<fsils_int>& diagPtr,
                               fsils_int nNo,
                               int block_size,
                               Array<double>& sparse_inverse)
{
  const auto& cfg = schur_sparsity_control();
  if (cfg.momentum_inverse_max_offdiag_per_row == -1 &&
      cfg.momentum_inverse_offdiag_score_min <= 0.0) {
    return;
  }

  const int block_entries = block_size * block_size;
  std::vector<double> diag_scale(static_cast<size_t>(nNo), 0.0);
  for (fsils_int row = 0; row < nNo; ++row) {
    diag_scale[static_cast<size_t>(row)] =
        std::max(block_max_abs(block_ptr(sparse_inverse, block_entries, diagPtr(row)), block_entries), 1e-30);
  }

  struct Candidate {
    fsils_int nz{-1};
    double score{0.0};
  };

  for (fsils_int row = 0; row < nNo; ++row) {
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(std::max<fsils_int>(0, rowPtr(1, row) - rowPtr(0, row))));

    for (fsils_int p = rowPtr(0, row); p <= rowPtr(1, row); ++p) {
      if (p == diagPtr(row)) {
        continue;
      }
      double* block = block_ptr(sparse_inverse, block_entries, p);
      const double norm = block_max_abs(block, block_entries);
      if (norm <= 1e-30) {
        set_zero(block, block_entries);
        continue;
      }

      const fsils_int col = colPtr(p);
      const double ref = std::sqrt(std::max(diag_scale[static_cast<size_t>(row)] *
                                            diag_scale[static_cast<size_t>(col)],
                                            1e-30));
      candidates.push_back(Candidate{p, norm / ref});
    }

    if (candidates.empty()) {
      continue;
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return a.score > b.score;
    });

    const int budget = (cfg.momentum_inverse_max_offdiag_per_row < 0)
                           ? static_cast<int>(candidates.size())
                           : std::min<int>(cfg.momentum_inverse_max_offdiag_per_row,
                                           static_cast<int>(candidates.size()));
    const int guaranteed = std::min<int>(cfg.guaranteed_offdiag_per_row, budget);

    int kept = 0;
    for (const auto& candidate : candidates) {
      const bool keep =
          (kept < guaranteed) ||
          (kept < budget && candidate.score >= cfg.momentum_inverse_offdiag_score_min);
      if (keep) {
        kept += 1;
        continue;
      }
      set_zero(block_ptr(sparse_inverse, block_entries, candidate.nz), block_entries);
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

void apply_point_block_inverse_transpose(const Array<double>& inv_blocks,
                                         int block_size,
                                         fsils_int nNo,
                                         Array<double>& x)
{
  std::vector<double> tmp(static_cast<size_t>(block_size), 0.0);
  for (fsils_int i = 0; i < nNo; ++i) {
    multiply_block_vector_transpose(block_ptr(inv_blocks, block_size * block_size, i), block_size, block_size,
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

void apply_block_ilu0_transpose(const Array<fsils_int>& rowPtr,
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
    multiply_block_vector_transpose(block_ptr(diag_inv, block_size * block_size, row),
                                    block_size, block_size,
                                    y.data() + static_cast<size_t>(row) * static_cast<size_t>(block_size),
                                    tmp.data());
    for (int k = 0; k < block_size; ++k) {
      y(k, row) = tmp[static_cast<size_t>(k)];
    }
    for (fsils_int p = diagPtr(row) + 1; p <= rowPtr(1, row); ++p) {
      subtract_transpose_block_vector_product(
          y.data() + static_cast<size_t>(colPtr(p)) * static_cast<size_t>(block_size),
          block_ptr(factors, block_size * block_size, p), block_size, block_size,
          tmp.data());
    }
  }

  for (fsils_int row = nNo; row-- > 0;) {
    for (int k = 0; k < block_size; ++k) {
      tmp[static_cast<size_t>(k)] = y(k, row);
    }
    for (fsils_int p = rowPtr(0, row); p < diagPtr(row); ++p) {
      subtract_transpose_block_vector_product(
          y.data() + static_cast<size_t>(colPtr(p)) * static_cast<size_t>(block_size),
          block_ptr(factors, block_size * block_size, p), block_size, block_size,
          tmp.data());
    }
  }

  x = y;
}

void apply_base_schur_preconditioner(const Array<fsils_int>& rowPtr,
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

void apply_momentum_hat(const Array<fsils_int>& rowPtr,
                        const Vector<fsils_int>& colPtr,
                        const Vector<fsils_int>& diagPtr,
                        int mom_ncomp,
                        fsils_int nNo,
                        const MomentumHatData& hat,
                        Array<double>& x)
{
  if (hat.use_ilu) {
    apply_block_ilu0(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat.ilu_factors, hat.ilu_diag_inv, x);
    return;
  }
  apply_point_block_inverse(hat.point_inv, mom_ncomp, nNo, x);
}

void apply_momentum_hat_transpose(const Array<fsils_int>& rowPtr,
                                  const Vector<fsils_int>& colPtr,
                                  const Vector<fsils_int>& diagPtr,
                                  int mom_ncomp,
                                  fsils_int nNo,
                                  const MomentumHatData& hat,
                                  Array<double>& x)
{
  if (hat.use_ilu) {
    apply_block_ilu0_transpose(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat.ilu_factors, hat.ilu_diag_inv, x);
    return;
  }
  apply_point_block_inverse_transpose(hat.point_inv, mom_ncomp, nNo, x);
}

[[nodiscard]] int reduced_local_component(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                          const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
                                          int full_component,
                                          int current_dof)
{
  const int system_dof = (lhs.system_dof > 0) ? lhs.system_dof : current_dof;
  return fe_fsi_linear_solver::fsils_reduced_local_component(update,
                                                             full_component,
                                                             current_dof,
                                                             system_dof);
}

bool fill_projected_reduced_vector(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& entries,
    int current_dof,
    Array<double>& values)
{
  values.resize(current_dof, lhs.nNo);
  values = 0.0;

  bool nonzero = false;
  for (const auto& entry : entries) {
    if (entry.node < 0 || entry.node >= lhs.nNo || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    const int comp = reduced_local_component(lhs, update, entry.full_component, current_dof);
    if (comp < 0 || comp >= current_dof) {
      continue;
    }
    values(comp, entry.node) += entry.value;
    nonzero = true;
  }
  return nonzero;
}

double sparse_dense_owned_dot_local(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& entries,
    const Array<double>& x,
    int current_dof)
{
  double local_dot = 0.0;
  for (const auto& entry : entries) {
    if (entry.node < 0 || entry.node >= lhs.mynNo || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    const int comp = reduced_local_component(lhs, update, entry.full_component, current_dof);
    if (comp < 0 || comp >= current_dof) {
      continue;
    }
    local_dot += entry.value * x(comp, entry.node);
  }
  return local_dot;
}

double dense_dense_owned_dot_local(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                   int dof,
                                   const Array<double>& left,
                                   const Array<double>& right)
{
  double local_dot = 0.0;
  for (fsils_int node = 0; node < lhs.mynNo; ++node) {
    for (int comp = 0; comp < dof; ++comp) {
      local_dot += left(comp, node) * right(comp, node);
    }
  }
  return local_dot;
}

void allreduce_sum_in_place(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                            std::vector<double>& values)
{
  if (lhs.commu.nTasks > 1 && !values.empty()) {
    fe_fsi_linear_solver::fsils_allreduce_sum_in_place(values.data(),
                                                       static_cast<int>(values.size()),
                                                       cm_mod::mpreal,
                                                       lhs.commu);
  }
}

void multiply_rect_transpose_local(const Array<fsils_int>& rowPtr,
                                   const Vector<fsils_int>& colPtr,
                                   fsils_int nNo,
                                   int out_dof,
                                   int in_dof,
                                   const Array<double>& K,
                                   const Array<double>& row_values,
                                   Array<double>& col_values)
{
  col_values.resize(in_dof, nNo);
  col_values = 0.0;

  std::vector<double> contrib(static_cast<size_t>(in_dof), 0.0);
  for (fsils_int row = 0; row < nNo; ++row) {
    const double* row_vec = row_values.data() + static_cast<size_t>(row) * static_cast<size_t>(out_dof);
    for (fsils_int p = rowPtr(0, row); p <= rowPtr(1, row); ++p) {
      const fsils_int col = colPtr(p);
      multiply_block_vector_transpose(block_ptr(K, out_dof * in_dof, p), out_dof, in_dof,
                                      row_vec, contrib.data());
      double* col_vec = col_values.data() + static_cast<size_t>(col) * static_cast<size_t>(in_dof);
      for (int comp = 0; comp < in_dof; ++comp) {
        col_vec[comp] += contrib[static_cast<size_t>(comp)];
      }
    }
  }
}

MomentumHatData build_momentum_hat_data(const Array<fsils_int>& rowPtr,
                                        const Vector<fsils_int>& colPtr,
                                        const Vector<fsils_int>& diagPtr,
                                        const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                        fsils_int nNo,
                                        int mom_ncomp,
                                        const Array<double>& K,
                                        fe_fsi_linear_solver::SchurMomentumApproximationType approx)
{
  using fe_fsi_linear_solver::SchurMomentumApproximationType;

  MomentumHatData hat{};
  Array<double> K_eff = K;
  const int block_entries = mom_ncomp * mom_ncomp;
  const int system_dof = (lhs.system_dof > 0) ? lhs.system_dof : mom_ncomp;

  auto component_index = [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
                             int full_comp) -> int {
    return fe_fsi_linear_solver::fsils_reduced_local_component(update,
                                                               full_comp,
                                                               mom_ncomp,
                                                               system_dof);
  };

  auto add_local_update_entries =
      [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update) {
    if (!update.active || std::abs(update.sigma) <= 1e-30) {
      return;
    }

    std::unordered_map<fsils_int, std::vector<std::pair<int, double>>> left_by_node;
    std::unordered_map<fsils_int, std::vector<std::pair<int, double>>> right_by_node;
    const auto& left_entries =
        !update.left_scaled_owned.empty() ? update.left_scaled_owned : update.left_owned;
    const auto& right_entries =
        !update.right_scaled_owned.empty() ? update.right_scaled_owned : update.right_owned;

    for (const auto& entry : left_entries) {
      if (entry.node < 0 || std::abs(entry.value) <= 1e-30) {
        continue;
      }
      const fsils_int node = entry.node;
      const int local_comp = component_index(update, entry.full_component);
      if (node < 0 || node >= nNo || local_comp < 0 || local_comp >= mom_ncomp) {
        continue;
      }
      left_by_node[node].emplace_back(local_comp, entry.value);
    }
    for (const auto& entry : right_entries) {
      if (entry.node < 0 || std::abs(entry.value) <= 1e-30) {
        continue;
      }
      const fsils_int node = entry.node;
      const int local_comp = component_index(update, entry.full_component);
      if (node < 0 || node >= nNo || local_comp < 0 || local_comp >= mom_ncomp) {
        continue;
      }
      right_by_node[node].emplace_back(local_comp, entry.value);
    }

    for (const auto& [node, left_vals] : left_by_node) {
      const auto it = right_by_node.find(node);
      if (it == right_by_node.end()) {
        continue;
      }
      const fsils_int diag_nz = diagPtr(node);
      double* diag_block = block_ptr(K_eff, block_entries, diag_nz);
      for (const auto& [li, lval] : left_vals) {
        for (const auto& [rj, rval] : it->second) {
          diag_block[li * mom_ncomp + rj] += update.sigma * lval * rval;
        }
      }
    }
  };

  for (const auto& update : lhs.reduced_updates) {
    add_local_update_entries(update);
  }

  switch (approx) {
    case SchurMomentumApproximationType::DIAG_K:
      build_point_inverse_blocks(diagPtr, mom_ncomp, nNo, K_eff, /*diagonal_only=*/true, hat.point_inv);
      build_sparse_block_diagonal_inverse(diagPtr, mom_ncomp, nNo, K_eff.ncols(), hat.point_inv,
                                          hat.sparse_inverse);
      return hat;
    case SchurMomentumApproximationType::BLOCKDIAG_K:
      build_point_inverse_blocks(diagPtr, mom_ncomp, nNo, K_eff, /*diagonal_only=*/false, hat.point_inv);
      build_sparse_block_diagonal_inverse(diagPtr, mom_ncomp, nNo, K_eff.ncols(), hat.point_inv,
                                          hat.sparse_inverse);
      return hat;
    case SchurMomentumApproximationType::ILU_K:
    case SchurMomentumApproximationType::ASM_K: {
      factorize_block_ilu0(rowPtr, colPtr, diagPtr, nNo, mom_ncomp, K_eff,
                           hat.ilu_factors, hat.ilu_diag_inv);
      hat.point_inv = hat.ilu_diag_inv;
      build_sparse_block_ilu_inverse(rowPtr, colPtr, diagPtr, nNo, K_eff.ncols(), mom_ncomp,
                                     hat.ilu_factors, hat.ilu_diag_inv, hat.sparse_inverse);
      prune_sparse_momentum_hat(rowPtr, colPtr, diagPtr, nNo, mom_ncomp, hat.sparse_inverse);
      hat.use_ilu = true;
      return hat;
    }
  }

  return hat;
}

void assemble_algebraic_schur(const Array<fsils_int>& rowPtr,
                              const Vector<fsils_int>& colPtr,
                              const Vector<fsils_int>& diagPtr,
                              fsils_int nNo,
                              int mom_ncomp,
                              int con_ncomp,
                              const Array<double>& D,
                              const Array<double>& G,
                              const Array<double>& L,
                              const Array<double>& momentum_hat_sparse,
                              Array<double>& shat)
{
  const auto& cfg = schur_sparsity_control();
  const int con_block_entries = con_ncomp * con_ncomp;
  const int mixed_block_entries = con_ncomp * mom_ncomp;
  const int mom_block_entries = mom_ncomp * mom_ncomp;
  shat = L;

  struct Candidate {
    fsils_int local_index{-1};
    double score{0.0};
  };

  std::vector<double> dh(static_cast<std::size_t>(mixed_block_entries), 0.0);
  std::vector<double> contrib(static_cast<std::size_t>(con_block_entries), 0.0);
  for (fsils_int row = 0; row < nNo; ++row) {
    const fsils_int row_begin = rowPtr(0, row);
    const fsils_int row_end = rowPtr(1, row);
    const fsils_int row_nnz = row_end - row_begin + 1;
    std::vector<double> row_corr(static_cast<std::size_t>(row_nnz) * static_cast<std::size_t>(con_block_entries), 0.0);

    for (fsils_int p_d = row_begin; p_d <= row_end; ++p_d) {
      const fsils_int alpha = colPtr(p_d);
      const double* d_block = block_ptr(D, mixed_block_entries, p_d);
      for (fsils_int p_h = rowPtr(0, alpha); p_h <= rowPtr(1, alpha); ++p_h) {
        const double* h_block = block_ptr(momentum_hat_sparse, mom_block_entries, p_h);
        if (block_is_zero(h_block, mom_block_entries)) {
          continue;
        }
        const fsils_int beta = colPtr(p_h);
        multiply_blocks(d_block, con_ncomp, mom_ncomp,
                        h_block, mom_ncomp,
                        dh.data());
        if (block_is_zero(dh.data(), mixed_block_entries)) {
          continue;
        }

        for (fsils_int p_g = rowPtr(0, beta); p_g <= rowPtr(1, beta); ++p_g) {
          const fsils_int col = colPtr(p_g);
          const fsils_int target = find_col_in_row(rowPtr, colPtr, row, col);
          if (target < 0) {
            continue;
          }
          multiply_blocks(dh.data(), con_ncomp, mom_ncomp,
                          block_ptr(G, mom_ncomp * con_ncomp, p_g), con_ncomp,
                          contrib.data());
          if (block_is_zero(contrib.data(), con_block_entries)) {
            continue;
          }

          double* target_block = row_corr.data() +
                                 static_cast<std::size_t>(target - row_begin) * static_cast<std::size_t>(con_block_entries);
          for (int e = 0; e < con_block_entries; ++e) {
            target_block[e] += contrib[static_cast<std::size_t>(e)];
          }
        }
      }
    }

    const fsils_int diag_local = diagPtr(row) - row_begin;
    const double diag_scale = std::max(block_max_abs(block_ptr(L, con_block_entries, diagPtr(row)),
                                                     con_block_entries),
                                       1e-30);
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(std::max<fsils_int>(0, row_nnz - 1)));
    for (fsils_int local = 0; local < row_nnz; ++local) {
      if (local == diag_local) {
        continue;
      }
      const double* corr_block = row_corr.data() +
                                 static_cast<std::size_t>(local) * static_cast<std::size_t>(con_block_entries);
      const double corr_norm = block_max_abs(corr_block, con_block_entries);
      if (corr_norm <= 1e-30) {
        continue;
      }
      const double l_scale = block_max_abs(block_ptr(L, con_block_entries, row_begin + local),
                                           con_block_entries);
      const double score = corr_norm / std::max({diag_scale, l_scale, 1e-30});
      candidates.push_back(Candidate{local, score});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return a.score > b.score;
    });

    const int budget = (cfg.shat_max_offdiag_per_row < 0)
                           ? static_cast<int>(candidates.size())
                           : std::min<int>(cfg.shat_max_offdiag_per_row, static_cast<int>(candidates.size()));
    const int guaranteed = std::min<int>(cfg.guaranteed_offdiag_per_row, budget);

    std::vector<unsigned char> keep(static_cast<std::size_t>(row_nnz), 0);
    keep[static_cast<std::size_t>(diag_local)] = 1;
    int kept = 0;
    for (const auto& candidate : candidates) {
      const bool retain =
          (kept < guaranteed) ||
          (kept < budget && candidate.score >= cfg.shat_row_score_min);
      if (!retain) {
        continue;
      }
      keep[static_cast<std::size_t>(candidate.local_index)] = 1;
      kept += 1;
    }

    for (fsils_int local = 0; local < row_nnz; ++local) {
      if (!keep[static_cast<std::size_t>(local)]) {
        continue;
      }
      double* target_block = block_ptr(shat, con_block_entries, row_begin + local);
      const double* corr_block = row_corr.data() +
                                 static_cast<std::size_t>(local) * static_cast<std::size_t>(con_block_entries);
      for (int e = 0; e < con_block_entries; ++e) {
        target_block[e] -= corr_block[static_cast<std::size_t>(e)];
      }
    }
  }
}

void build_reduced_schur_correction(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                    int mom_ncomp,
                                    int con_ncomp,
                                    const Array<double>& D,
                                    const Array<double>& G,
                                    const MomentumHatData& momentum_hat,
                                    SchurPreconditionerData& pc)
{
  struct ReducedColumn {
    const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType* update{nullptr};
    Array<double> momentum_left_hat;
    Array<double> schur_left;
    Array<double> schur_right;
  };

  std::vector<ReducedColumn> columns;
  auto append_dense_seed_block = [](std::vector<double>& dense_seed,
                                    int old_rank,
                                    const std::vector<double>& block,
                                    int block_rank) {
    std::vector<double> expanded(static_cast<std::size_t>(old_rank + block_rank) *
                                     static_cast<std::size_t>(old_rank + block_rank),
                                 0.0);
    for (int i = 0; i < old_rank; ++i) {
      for (int j = 0; j < old_rank; ++j) {
        expanded[static_cast<std::size_t>(i) * static_cast<std::size_t>(old_rank + block_rank) +
                 static_cast<std::size_t>(j)] =
            dense_seed[static_cast<std::size_t>(i) * static_cast<std::size_t>(old_rank) +
                       static_cast<std::size_t>(j)];
      }
    }
    for (int i = 0; i < block_rank; ++i) {
      for (int j = 0; j < block_rank; ++j) {
        expanded[static_cast<std::size_t>(old_rank + i) * static_cast<std::size_t>(old_rank + block_rank) +
                 static_cast<std::size_t>(old_rank + j)] =
            block[static_cast<std::size_t>(i) * static_cast<std::size_t>(block_rank) +
                  static_cast<std::size_t>(j)];
      }
    }
    dense_seed.swap(expanded);
  };

  auto append_column = [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update) {
    if (!update.active) {
      return;
    }

    const auto& left_entries = !update.left_scaled.empty() ? update.left_scaled : update.left;
    const auto& right_entries = !update.right_scaled.empty() ? update.right_scaled : update.right;
    if (left_entries.empty() || right_entries.empty()) {
      return;
    }

    ReducedColumn col;
    col.update = &update;
    if (!fill_projected_reduced_vector(lhs, update, left_entries, mom_ncomp, col.momentum_left_hat)) {
      return;
    }

    Array<double> momentum_right_t;
    if (!fill_projected_reduced_vector(lhs, update, right_entries, mom_ncomp, momentum_right_t)) {
      return;
    }

    apply_momentum_hat(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                       mom_ncomp, lhs.nNo, momentum_hat, col.momentum_left_hat);
    apply_momentum_hat_transpose(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                 mom_ncomp, lhs.nNo, momentum_hat, momentum_right_t);

    col.schur_left.resize(con_ncomp, lhs.nNo);
    spar_mul::fsils_spar_mul_rect(lhs, lhs.rowPtr, lhs.colPtr,
                                  con_ncomp, mom_ncomp, D,
                                  col.momentum_left_hat, col.schur_left);
    multiply_rect_transpose_local(lhs.rowPtr, lhs.colPtr, lhs.nNo,
                                  mom_ncomp, con_ncomp, G,
                                  momentum_right_t, col.schur_right);
    columns.push_back(std::move(col));
  };

  std::vector<double> dense_seed;
  dense_seed.reserve(lhs.reduced_updates.size());

  const bool has_grouped_bordered = !lhs.grouped_bordered_field_couplings.empty();
  for (const auto& update : lhs.reduced_updates) {
    if (!update.active || std::abs(update.sigma) <= 1e-30) {
      continue;
    }
    if (has_grouped_bordered && update.grouped_coupling_id >= 0) {
      continue;
    }

    const int old_rank = static_cast<int>(columns.size());
    append_column(update);
    if (static_cast<int>(columns.size()) != old_rank + 1) {
      continue;
    }
    append_dense_seed_block(dense_seed, old_rank,
                            std::vector<double>{safe_inverse(columns.back().update->sigma)},
                            /*block_rank=*/1);
  }

  for (const auto& group : lhs.grouped_bordered_field_couplings) {
    if (!group.active || group.modes.empty()) {
      continue;
    }
    const int block_rank = static_cast<int>(group.modes.size());
    if (group.aux_matrix.size() != static_cast<std::size_t>(block_rank * block_rank)) {
      continue;
    }

    const int old_rank = static_cast<int>(columns.size());
    for (const auto& mode : group.modes) {
      append_column(mode);
    }
    if (static_cast<int>(columns.size()) != old_rank + block_rank) {
      columns.resize(static_cast<std::size_t>(old_rank));
      continue;
    }

    std::vector<double> seed_block(group.aux_matrix.size(), 0.0);
    for (std::size_t idx = 0; idx < group.aux_matrix.size(); ++idx) {
      seed_block[idx] = -group.aux_matrix[idx];
    }
    append_dense_seed_block(dense_seed, old_rank, seed_block, block_rank);
  }

  const int rank = static_cast<int>(columns.size());
  if (rank == 0 || dense_seed.size() != static_cast<std::size_t>(rank * rank)) {
    return;
  }

  std::vector<double> dense_m = dense_seed;
  for (int i = 0; i < rank; ++i) {
    const auto& update_i = *columns[static_cast<std::size_t>(i)].update;
    const auto& right_owned =
        !update_i.right_scaled_owned.empty() ? update_i.right_scaled_owned : update_i.right_owned;
    for (int j = 0; j < rank; ++j) {
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(j)] +=
          sparse_dense_owned_dot_local(lhs, update_i, right_owned,
                                       columns[static_cast<std::size_t>(j)].momentum_left_hat,
                                       mom_ncomp);
    }
  }
  allreduce_sum_in_place(lhs, dense_m);

  pc.low_rank_right.clear();
  pc.low_rank_preconditioned_left.clear();
  pc.low_rank_right.reserve(columns.size());
  pc.low_rank_preconditioned_left.reserve(columns.size());
  for (auto& column : columns) {
    pc.low_rank_right.push_back(column.schur_right);
    Array<double> z = column.schur_left;
    apply_base_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                    pc, con_ncomp, lhs.nNo, z);
    pc.low_rank_preconditioned_left.push_back(std::move(z));
  }

  std::vector<double> dense_ctz(static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    for (int j = 0; j < rank; ++j) {
      dense_ctz[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(j)] =
          dense_dense_owned_dot_local(lhs, con_ncomp,
                                      pc.low_rank_right[static_cast<std::size_t>(i)],
                                      pc.low_rank_preconditioned_left[static_cast<std::size_t>(j)]);
    }
  }
  allreduce_sum_in_place(lhs, dense_ctz);

  for (std::size_t idx = 0; idx < dense_m.size(); ++idx) {
    dense_m[idx] += dense_ctz[idx];
  }

  pc.low_rank_inner_inv.assign(dense_m.size(), 0.0);
  if (!invert_dense_block(rank, dense_m.data(), pc.low_rank_inner_inv.data())) {
    pc.low_rank_right.clear();
    pc.low_rank_preconditioned_left.clear();
    pc.low_rank_inner_inv.clear();
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
    const auto& cfg = schur_sparsity_control();
    const MomentumHatData momentum_hat =
        build_momentum_hat_data(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, lhs.nNo,
                                mom_ncomp, K, ls.schur_momentum_approximation);

    Array<double> shat(L.nrows(), L.ncols());
    assemble_algebraic_schur(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, mom_ncomp, con_ncomp,
                             D, G, L, momentum_hat.sparse_inverse, shat);
    factorize_block_ilu0(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, con_ncomp, shat,
                         pc.ilu_factors, pc.ilu_diag_inv);
    pc.use_ilu = true;
    pc.momentum_hat = momentum_hat;
    pc.operator_D = &D;
    pc.operator_G = &G;
    pc.operator_L_storage = L;
    pc.operator_L = &pc.operator_L_storage;
    pc.operator_mom_ncomp = mom_ncomp;
    pc.operator_con_ncomp = con_ncomp;
    pc.operator_refinement_steps = cfg.operator_refinement_steps;
    pc.operator_refinement_omega = cfg.operator_refinement_omega;

    if (!lhs.reduced_updates.empty()) {
      build_reduced_schur_correction(lhs, mom_ncomp, con_ncomp, D, G, momentum_hat, pc);
    }
  }

  return pc;
}

void apply_schur_low_rank_correction(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                     const SchurPreconditionerData& pc,
                                     int con_ncomp,
                                     fsils_int nNo,
                                     Array<double>& x)
{
  const int rank = static_cast<int>(pc.low_rank_right.size());
  if (rank == 0 || pc.low_rank_preconditioned_left.size() != pc.low_rank_right.size() ||
      pc.low_rank_inner_inv.size() != static_cast<std::size_t>(rank * rank)) {
    return;
  }

  std::vector<double> gamma(static_cast<size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    gamma[static_cast<size_t>(i)] =
        dense_dense_owned_dot_local(lhs, con_ncomp,
                                    pc.low_rank_right[static_cast<size_t>(i)], x);
  }
  allreduce_sum_in_place(lhs, gamma);

  std::vector<double> delta(static_cast<size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    double sum = 0.0;
    for (int j = 0; j < rank; ++j) {
      sum += pc.low_rank_inner_inv[static_cast<size_t>(i) * static_cast<size_t>(rank) + static_cast<size_t>(j)] *
             gamma[static_cast<size_t>(j)];
    }
    delta[static_cast<size_t>(i)] = sum;
  }

  for (int i = 0; i < rank; ++i) {
    const double scale = delta[static_cast<size_t>(i)];
    if (std::abs(scale) <= 1e-30) {
      continue;
    }
    const auto& z = pc.low_rank_preconditioned_left[static_cast<size_t>(i)];
    for (fsils_int node = 0; node < nNo; ++node) {
      for (int comp = 0; comp < con_ncomp; ++comp) {
        x(comp, node) -= z(comp, node) * scale;
      }
    }
  }
}

void apply_hat_schur_operator(const Array<fsils_int>& rowPtr,
                              const Vector<fsils_int>& colPtr,
                              const Vector<fsils_int>& diagPtr,
                              fe_fsi_linear_solver::FSILS_lhsType& lhs,
                              SchurPreconditionerData& pc,
                              const Array<double>& in_vec,
                              Array<double>& out_vec)
{
  const int mom_ncomp = pc.operator_mom_ncomp;
  const int con_ncomp = pc.operator_con_ncomp;
  const fsils_int nNo = lhs.nNo;
  if (pc.operator_D == nullptr || pc.operator_G == nullptr || pc.operator_L == nullptr ||
      mom_ncomp <= 0 || con_ncomp <= 0) {
    out_vec = in_vec;
    return;
  }

  pc.scratch_gp.resize(mom_ncomp, nNo);
  pc.scratch_hgp.resize(mom_ncomp, nNo);
  pc.scratch_sp.resize(con_ncomp, nNo);
  pc.scratch_dgp.resize(con_ncomp, nNo);

  spar_mul::fsils_spar_mul_rect_vv_fused(lhs, rowPtr, colPtr,
      mom_ncomp, con_ncomp, *pc.operator_G, *pc.operator_L, in_vec, pc.scratch_gp, pc.scratch_sp);

  pc.scratch_hgp = pc.scratch_gp;
  apply_momentum_hat(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, pc.momentum_hat, pc.scratch_hgp);

  spar_mul::fsils_spar_mul_rect(lhs, rowPtr, colPtr,
      con_ncomp, mom_ncomp, *pc.operator_D, pc.scratch_hgp, pc.scratch_dgp);

  out_vec.resize(con_ncomp, nNo);
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; ++i) {
    for (int k = 0; k < con_ncomp; ++k) {
      out_vec(k, i) = pc.scratch_sp(k, i) - pc.scratch_dgp(k, i);
    }
  }
}

void apply_schur_preconditioner(const Array<fsils_int>& rowPtr,
                                const Vector<fsils_int>& colPtr,
                                const Vector<fsils_int>& diagPtr,
                                fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                SchurPreconditionerData& pc,
                                int con_ncomp,
                                fsils_int nNo,
                                Array<double>& x)
{
  if (pc.operator_refinement_steps > 0 && pc.operator_D != nullptr &&
      pc.operator_G != nullptr && pc.operator_L != nullptr) {
    pc.scratch_rhs = x;
  }

  apply_base_schur_preconditioner(rowPtr, colPtr, diagPtr, pc, con_ncomp, nNo, x);
  apply_schur_low_rank_correction(lhs, pc, con_ncomp, nNo, x);

  for (int step = 0; step < pc.operator_refinement_steps; ++step) {
    apply_hat_schur_operator(rowPtr, colPtr, diagPtr, lhs, pc, x, pc.scratch_ax);

    pc.scratch_correction = pc.scratch_rhs;
    #pragma omp parallel for schedule(static)
    for (fsils_int node = 0; node < nNo; ++node) {
      for (int comp = 0; comp < con_ncomp; ++comp) {
        pc.scratch_correction(comp, node) -= pc.scratch_ax(comp, node);
      }
    }

    apply_base_schur_preconditioner(rowPtr, colPtr, diagPtr, pc, con_ncomp, nNo, pc.scratch_correction);
    apply_schur_low_rank_correction(lhs, pc, con_ncomp, nNo, pc.scratch_correction);

    const double omega = pc.operator_refinement_omega;
    #pragma omp parallel for schedule(static)
    for (fsils_int node = 0; node < nNo; ++node) {
      for (int comp = 0; comp < con_ncomp; ++comp) {
        x(comp, node) += omega * pc.scratch_correction(comp, node);
      }
    }
  }
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

  auto& cache = schur_cache_entry(ls);
  double pc_setup_time = 0.0;
  if (!cache.valid || cache.mom_ncomp != mom_ncomp || cache.con_ncomp != con_ncomp || cache.nNo != nNo) {
    const double setup_t0 = fe_fsi_linear_solver::fsils_cpu_t();
    cache.preconditioner = build_schur_preconditioner(lhs, ls, mom_ncomp, con_ncomp, K, D, G, L);
    cache.preconditioner.operator_L = &cache.preconditioner.operator_L_storage;
    pc_setup_time = fe_fsi_linear_solver::fsils_cpu_t() - setup_t0;
    cache.valid = true;
    cache.mom_ncomp = mom_ncomp;
    cache.con_ncomp = con_ncomp;
    cache.nNo = nNo;
    cache.previous_solution.resize(0, 0);
  }
  auto& pc = cache.preconditioner;

  auto apply_exact_schur_operator = [&](const Array<double>& in_vec, Array<double>& out_vec) {
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
  };

  auto apply_schur_operator = [&](const Array<double>& in_vec, Array<double>& out_vec) {
    apply_exact_schur_operator(in_vec, out_vec);
    apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, pc, con_ncomp, nNo, out_vec);
  };

  const Array<double> rhs = R;
  Array<double> rhs_preconditioned = rhs;
  apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, pc, con_ncomp, nNo, rhs_preconditioned);

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;

  const double rhs_norm = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, rhs_preconditioned);
  ls.iNorm = rhs_norm;
  const double eps = std::max(ls.absTol, ls.relTol * rhs_norm);

  X = 0.0;
  R = rhs_preconditioned;

  double err = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, R);
  double errO = std::max(err, rhs_norm);
  double rho = err * err;
  double beta = rho;

  P = R;
  Rh = R;
  int i_itr = 1;

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

void reset_schur_cache(fe_fsi_linear_solver::FSILS_subLsType& ls)
{
  schur_cache_registry().erase(&ls);
}

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
