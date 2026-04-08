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
#include "block_schur_strategy_selector.h"
#include "bcast.h"
#include "distributed_mpi_ops.h"
#include "distributed_low_rank_correction.h"
#include "distributed_sparse_operator.h"
#include "dot.h"
#include "gmres.h"
#include "norm.h"
#include "omp_la.h"

#include <unordered_map>

#include "Array3.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>

namespace bicgs {

namespace dso = fe_fsi_linear_solver::distributed_sparse_operator;
namespace dsb = fe_fsi_linear_solver::distributed_solver_bundles;

constexpr int kNativeFaceDuplicateCouplingId = -2;

/// @brief Biconjugate-gradient stabilized algorithm for vector systems.
void bicgsv(const fe_fsi_linear_solver::distributed_solver_bundles::VectorLinearSystem& system,
    fe_fsi_linear_solver::FSILS_subLsType& ls, Array<double>& R)
{
  #define n_debug_bicgsv
  #ifdef debug_bicgsv
  DebugMsg dmsg(__func__,  system.lhs->commu.task);
  dmsg.banner();
  #endif

  using namespace fe_fsi_linear_solver;
  auto& lhs = *system.lhs;
  const int dof = system.components;
  const auto& A = system.A;

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

    A.apply(
        dso::ghost_synced_input(dof, P),
        dso::ghost_synced_output(dof, V));
    double denom_alpha = dot::fsils_dot_v(dof, mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_v(dof, nNo, S, R, -alpha, V);

    A.apply(
        dso::ghost_synced_input(dof, S),
        dso::ghost_synced_output(dof, T));
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

void bicgsv(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Array<double>& K, Array<double>& R)
{
  bicgsv(fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, dof, K), ls, R);
}

//--------
// bicgss
//--------
//
void bicgss(const fe_fsi_linear_solver::distributed_solver_bundles::ScalarLinearSystem& system,
    fe_fsi_linear_solver::FSILS_subLsType& ls, Vector<double>& R)
{
  #define n_debug_bicgss
  #ifdef debug_bicgss
  DebugMsg dmsg(__func__,  system.lhs->commu.task);
  dmsg.banner();
  #endif

  using namespace fe_fsi_linear_solver;
  auto& lhs = *system.lhs;
  const auto& A = system.A;

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

    A.apply(
        dso::ghost_synced_input(P),
        dso::ghost_synced_output(V));
    double denom_alpha = dot::fsils_dot_s(mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_s(nNo, S, R, -alpha, V);

    A.apply(
        dso::ghost_synced_input(S),
        dso::ghost_synced_output(T));
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

void bicgss(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls,
            const Vector<double>& K, Vector<double>& R)
{
  bicgss(fe_fsi_linear_solver::distributed_solver_bundles::make_scalar_linear_system(lhs, K), ls, R);
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

[[nodiscard]] bool skip_post_solve_overlap_sum() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_SKIP_POST_SOLVE_COMM");
  if (!env) {
    return false;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  if (*env == '\0') {
    return false;
  }
  return *env != '0';
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
  bool use_asm{false};
  fsils_int mynNo{0};
  const fe_fsi_linear_solver::FSILS_commuType* commu{nullptr};
  std::vector<Array<double>> low_rank_right;
  std::vector<Array<double>> low_rank_preconditioned_left;
  std::vector<Array<double>> low_rank_left;
  std::vector<Array<double>> low_rank_preconditioned_right_t;
  std::vector<double> low_rank_inner_inv;
  std::vector<double> low_rank_inner_inv_t;
  std::vector<fsils_int> asm_patch_ptr;
  std::vector<fsils_int> asm_patch_nodes;
  std::vector<fsils_int> asm_patch_matrix_ptr;
  std::vector<double> asm_patch_inverse;
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
  fe_fsi_linear_solver::distributed_low_rank_correction::DistributedLowRankCorrection
      explicit_low_rank_correction;
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

[[nodiscard]] inline const double* block_ptr(const Array<double>& A, int block_entries, fsils_int index)
{
  return A.data() + static_cast<size_t>(block_entries) * static_cast<size_t>(index);
}

[[nodiscard]] inline double* block_ptr(Array<double>& A, int block_entries, fsils_int index)
{
  return A.data() + static_cast<size_t>(block_entries) * static_cast<size_t>(index);
}

void dense_axpy(int dof,
                fsils_int nNo,
                const Array<double>& src,
                double scale,
                Array<double>& dst);

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

void build_zero_overlap_asm_patches(const Array<fsils_int>& rowPtr,
                                    const Vector<fsils_int>& colPtr,
                                    fsils_int nNo,
                                    std::vector<fsils_int>& patch_ptr,
                                    std::vector<fsils_int>& patch_nodes)
{
  patch_ptr.clear();
  patch_nodes.clear();
  patch_ptr.push_back(0);

  std::vector<unsigned char> assigned(static_cast<std::size_t>(nNo), 0);
  std::vector<fsils_int> patch;
  for (fsils_int row = 0; row < nNo; ++row) {
    if (assigned[static_cast<std::size_t>(row)]) {
      continue;
    }

    patch.clear();
    patch.push_back(row);
    assigned[static_cast<std::size_t>(row)] = 1;

    for (fsils_int p = rowPtr(0, row); p <= rowPtr(1, row); ++p) {
      const fsils_int col = colPtr(p);
      if (col < 0 || col >= nNo || assigned[static_cast<std::size_t>(col)]) {
        continue;
      }
      patch.push_back(col);
      assigned[static_cast<std::size_t>(col)] = 1;
    }

    std::sort(patch.begin(), patch.end());
    patch_nodes.insert(patch_nodes.end(), patch.begin(), patch.end());
    patch_ptr.push_back(static_cast<fsils_int>(patch_nodes.size()));
  }
}

void build_zero_overlap_asm_inverse(const Array<fsils_int>& rowPtr,
                                    const Vector<fsils_int>& colPtr,
                                    fsils_int nNo,
                                    fsils_int nnz,
                                    int block_size,
                                    const Array<double>& values,
                                    std::vector<fsils_int>& patch_ptr,
                                    std::vector<fsils_int>& patch_nodes,
                                    std::vector<fsils_int>& patch_matrix_ptr,
                                    std::vector<double>& patch_inverse,
                                    Array<double>& sparse_inverse)
{
  build_zero_overlap_asm_patches(rowPtr, colPtr, nNo, patch_ptr, patch_nodes);

  patch_matrix_ptr.clear();
  patch_inverse.clear();
  patch_matrix_ptr.push_back(0);
  sparse_inverse.resize(block_size * block_size, nnz);
  sparse_inverse = 0.0;

  std::vector<double> dense;
  std::vector<double> dense_inv;
  std::vector<double> fallback_diag;
  std::vector<double> fallback_inv;
  for (std::size_t patch_index = 0; patch_index + 1 < patch_ptr.size(); ++patch_index) {
    const fsils_int begin = patch_ptr[patch_index];
    const fsils_int end = patch_ptr[patch_index + 1];
    const int patch_size = static_cast<int>(end - begin);
    if (patch_size <= 0) {
      patch_matrix_ptr.push_back(static_cast<fsils_int>(patch_inverse.size()));
      continue;
    }

    const int patch_dof = patch_size * block_size;
    dense.assign(static_cast<std::size_t>(patch_dof) * static_cast<std::size_t>(patch_dof), 0.0);
    dense_inv.assign(static_cast<std::size_t>(patch_dof) * static_cast<std::size_t>(patch_dof), 0.0);
    fallback_diag.assign(static_cast<std::size_t>(block_size) * static_cast<std::size_t>(block_size), 0.0);
    fallback_inv.assign(static_cast<std::size_t>(block_size) * static_cast<std::size_t>(block_size), 0.0);

    for (int local_row = 0; local_row < patch_size; ++local_row) {
      const fsils_int row_node = patch_nodes[static_cast<std::size_t>(begin + local_row)];
      for (int local_col = 0; local_col < patch_size; ++local_col) {
        const fsils_int col_node = patch_nodes[static_cast<std::size_t>(begin + local_col)];
        const fsils_int nz = find_col_in_row(rowPtr, colPtr, row_node, col_node);
        if (nz < 0) {
          continue;
        }

        const double* block = block_ptr(values, block_size * block_size, nz);
        for (int bi = 0; bi < block_size; ++bi) {
          for (int bj = 0; bj < block_size; ++bj) {
            dense[static_cast<std::size_t>(local_row * block_size + bi) *
                      static_cast<std::size_t>(patch_dof) +
                  static_cast<std::size_t>(local_col * block_size + bj)] =
                block[bi * block_size + bj];
          }
        }
      }
    }

    if (!invert_dense_block(patch_dof, dense.data(), dense_inv.data())) {
      std::fill(dense_inv.begin(), dense_inv.end(), 0.0);
      for (int local_row = 0; local_row < patch_size; ++local_row) {
        const fsils_int row_node = patch_nodes[static_cast<std::size_t>(begin + local_row)];
        const fsils_int nz = find_col_in_row(rowPtr, colPtr, row_node, row_node);
        if (nz < 0) {
          continue;
        }
        const double* block = block_ptr(values, block_size * block_size, nz);
        std::copy(block, block + block_size * block_size, fallback_diag.begin());
        if (!invert_dense_block(block_size, fallback_diag.data(), fallback_inv.data())) {
          for (int d = 0; d < block_size; ++d) {
            dense_inv[static_cast<std::size_t>(local_row * block_size + d) *
                          static_cast<std::size_t>(patch_dof) +
                      static_cast<std::size_t>(local_row * block_size + d)] =
                safe_inverse(block[d * block_size + d]);
          }
        } else {
          for (int bi = 0; bi < block_size; ++bi) {
            for (int bj = 0; bj < block_size; ++bj) {
              dense_inv[static_cast<std::size_t>(local_row * block_size + bi) *
                            static_cast<std::size_t>(patch_dof) +
                        static_cast<std::size_t>(local_row * block_size + bj)] =
                  fallback_inv[static_cast<std::size_t>(bi) * static_cast<std::size_t>(block_size) +
                               static_cast<std::size_t>(bj)];
            }
          }
        }
      }
    }

    for (int local_row = 0; local_row < patch_size; ++local_row) {
      const fsils_int row_node = patch_nodes[static_cast<std::size_t>(begin + local_row)];
      for (int local_col = 0; local_col < patch_size; ++local_col) {
        const fsils_int col_node = patch_nodes[static_cast<std::size_t>(begin + local_col)];
        const fsils_int nz = find_col_in_row(rowPtr, colPtr, row_node, col_node);
        if (nz < 0) {
          continue;
        }

        double* out = block_ptr(sparse_inverse, block_size * block_size, nz);
        for (int bi = 0; bi < block_size; ++bi) {
          for (int bj = 0; bj < block_size; ++bj) {
            out[bi * block_size + bj] =
                dense_inv[static_cast<std::size_t>(local_row * block_size + bi) *
                              static_cast<std::size_t>(patch_dof) +
                          static_cast<std::size_t>(local_col * block_size + bj)];
          }
        }
      }
    }

    patch_matrix_ptr.push_back(static_cast<fsils_int>(patch_inverse.size() + dense_inv.size()));
    patch_inverse.insert(patch_inverse.end(), dense_inv.begin(), dense_inv.end());
  }
}

void apply_zero_overlap_asm(int block_size,
                            const std::vector<fsils_int>& patch_ptr,
                            const std::vector<fsils_int>& patch_nodes,
                            const std::vector<fsils_int>& patch_matrix_ptr,
                            const std::vector<double>& patch_inverse,
                            Array<double>& x,
                            bool transpose)
{
  const fsils_int nNo = x.ncols();
  Array<double> y(block_size, nNo);
  y = 0.0;

  std::vector<double> patch_rhs;
  std::vector<double> patch_sol;
  for (std::size_t patch_index = 0; patch_index + 1 < patch_ptr.size(); ++patch_index) {
    const fsils_int begin = patch_ptr[patch_index];
    const fsils_int end = patch_ptr[patch_index + 1];
    const int patch_size = static_cast<int>(end - begin);
    if (patch_size <= 0) {
      continue;
    }
    const int patch_dof = patch_size * block_size;
    const double* inv =
        patch_inverse.data() + static_cast<std::size_t>(patch_matrix_ptr[patch_index]);

    patch_rhs.assign(static_cast<std::size_t>(patch_dof), 0.0);
    patch_sol.assign(static_cast<std::size_t>(patch_dof), 0.0);
    for (int local_node = 0; local_node < patch_size; ++local_node) {
      const fsils_int node = patch_nodes[static_cast<std::size_t>(begin + local_node)];
      for (int comp = 0; comp < block_size; ++comp) {
        patch_rhs[static_cast<std::size_t>(local_node * block_size + comp)] = x(comp, node);
      }
    }

    if (transpose) {
      multiply_block_vector_transpose(inv, patch_dof, patch_dof, patch_rhs.data(), patch_sol.data());
    } else {
      multiply_block_vector(inv, patch_dof, patch_dof, patch_rhs.data(), patch_sol.data());
    }

    for (int local_node = 0; local_node < patch_size; ++local_node) {
      const fsils_int node = patch_nodes[static_cast<std::size_t>(begin + local_node)];
      for (int comp = 0; comp < block_size; ++comp) {
        y(comp, node) = patch_sol[static_cast<std::size_t>(local_node * block_size + comp)];
      }
    }
  }

  x = y;
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

void sync_schur_overlap(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                        int con_ncomp,
                        Array<double>& x)
{
  const fe_fsi_linear_solver::HaloExchange halo(lhs);
  halo.sync_vector(con_ncomp, x, skip_post_solve_overlap_sum());
}

void apply_momentum_hat_low_rank_correction(int mom_ncomp,
                                            const MomentumHatData& hat,
                                            bool transpose,
                                            Array<double>& x)
{
  const auto& right = transpose ? hat.low_rank_left : hat.low_rank_right;
  const auto& preconditioned_left =
      transpose ? hat.low_rank_preconditioned_right_t : hat.low_rank_preconditioned_left;
  const auto& inner_inv =
      transpose ? hat.low_rank_inner_inv_t : hat.low_rank_inner_inv;
  const int rank = static_cast<int>(right.size());
  if (rank <= 0 || preconditioned_left.size() != right.size() ||
      inner_inv.size() != static_cast<std::size_t>(rank * rank) ||
      hat.commu == nullptr) {
    return;
  }

  std::vector<double> rhs(static_cast<std::size_t>(rank), 0.0);
  std::vector<double> alpha(static_cast<std::size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    rhs[static_cast<std::size_t>(i)] =
        dot::fsils_dot_v(mom_ncomp,
                         hat.mynNo,
                         const_cast<fe_fsi_linear_solver::FSILS_commuType&>(*hat.commu),
                         right[static_cast<std::size_t>(i)],
                         x);
  }

  for (int i = 0; i < rank; ++i) {
    double value = 0.0;
    for (int j = 0; j < rank; ++j) {
      value += inner_inv[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                         static_cast<std::size_t>(j)] *
               rhs[static_cast<std::size_t>(j)];
    }
    alpha[static_cast<std::size_t>(i)] = value;
  }

  for (int i = 0; i < rank; ++i) {
    dense_axpy(mom_ncomp,
               x.ncols(),
               preconditioned_left[static_cast<std::size_t>(i)],
               -alpha[static_cast<std::size_t>(i)],
               x);
  }
}

void apply_momentum_hat(const Array<fsils_int>& rowPtr,
                        const Vector<fsils_int>& colPtr,
                        const Vector<fsils_int>& diagPtr,
                        int mom_ncomp,
                        fsils_int nNo,
                        const MomentumHatData& hat,
                        Array<double>& x)
{
  if (hat.use_asm) {
    apply_zero_overlap_asm(mom_ncomp,
                           hat.asm_patch_ptr,
                           hat.asm_patch_nodes,
                           hat.asm_patch_matrix_ptr,
                           hat.asm_patch_inverse,
                           x,
                           /*transpose=*/false);
    apply_momentum_hat_low_rank_correction(mom_ncomp, hat, /*transpose=*/false, x);
    return;
  }
  if (hat.use_ilu) {
    apply_block_ilu0(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat.ilu_factors, hat.ilu_diag_inv, x);
    apply_momentum_hat_low_rank_correction(mom_ncomp, hat, /*transpose=*/false, x);
    return;
  }
  apply_point_block_inverse(hat.point_inv, mom_ncomp, nNo, x);
  apply_momentum_hat_low_rank_correction(mom_ncomp, hat, /*transpose=*/false, x);
}

void apply_momentum_hat_transpose(const Array<fsils_int>& rowPtr,
                                  const Vector<fsils_int>& colPtr,
                                  const Vector<fsils_int>& diagPtr,
                                  int mom_ncomp,
                                  fsils_int nNo,
                                  const MomentumHatData& hat,
                                  Array<double>& x)
{
  if (hat.use_asm) {
    apply_zero_overlap_asm(mom_ncomp,
                           hat.asm_patch_ptr,
                           hat.asm_patch_nodes,
                           hat.asm_patch_matrix_ptr,
                           hat.asm_patch_inverse,
                           x,
                           /*transpose=*/true);
    apply_momentum_hat_low_rank_correction(mom_ncomp, hat, /*transpose=*/true, x);
    return;
  }
  if (hat.use_ilu) {
    apply_block_ilu0_transpose(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat.ilu_factors, hat.ilu_diag_inv, x);
    apply_momentum_hat_low_rank_correction(mom_ncomp, hat, /*transpose=*/true, x);
    return;
  }
  apply_point_block_inverse_transpose(hat.point_inv, mom_ncomp, nNo, x);
  apply_momentum_hat_low_rank_correction(mom_ncomp, hat, /*transpose=*/true, x);
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

bool fill_projected_face_vector(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                const fe_fsi_linear_solver::FSILS_faceType& face,
                                int current_dof,
                                Array<double>& values)
{
  values.resize(current_dof, lhs.nNo);
  values = 0.0;

  const int face_dof = std::min(face.dof, current_dof);
  bool nonzero = false;
  for (int a = 0; a < face.nNo; ++a) {
    const fsils_int node = face.glob(a);
    if (node < 0 || node >= lhs.nNo) {
      continue;
    }
    for (int comp = 0; comp < face_dof; ++comp) {
      const double value = face.valM(comp, a);
      if (std::abs(value) <= 1e-30) {
        continue;
      }
      values(comp, node) += value;
      nonzero = true;
    }
  }
  return nonzero;
}

bool fill_projected_block_vector(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& entries,
    int block_start,
    int block_dof,
    Array<double>& values)
{
  values.resize(block_dof, lhs.nNo);
  values = 0.0;

  bool nonzero = false;
  for (const auto& entry : entries) {
    if (entry.node < 0 || entry.node >= lhs.nNo || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    const int comp = entry.full_component - block_start;
    if (comp < 0 || comp >= block_dof) {
      continue;
    }
    values(comp, entry.node) += entry.value;
    nonzero = true;
  }
  return nonzero;
}

bool entries_touch_constraint_block(
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& entries,
    int mom_ncomp,
    int con_ncomp)
{
  const int con_begin = mom_ncomp;
  const int con_end = mom_ncomp + con_ncomp;
  for (const auto& entry : entries) {
    if (entry.node < 0 || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    if (entry.full_component >= con_begin && entry.full_component < con_end) {
      return true;
    }
  }
  return false;
}

bool reduced_updates_touch_constraint_block(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    int mom_ncomp,
    int con_ncomp)
{
  bool local_touch = false;
  for (const auto& update : lhs.reduced_updates) {
    if (!update.active || std::abs(update.sigma) <= 1e-30) {
      continue;
    }
    const auto& left_entries = !update.left_scaled.empty() ? update.left_scaled : update.left;
    const auto& right_entries = !update.right_scaled.empty() ? update.right_scaled : update.right;
    if (entries_touch_constraint_block(left_entries, mom_ncomp, con_ncomp) ||
        entries_touch_constraint_block(right_entries, mom_ncomp, con_ncomp)) {
      local_touch = true;
      break;
    }
  }
  int local_touch_int = local_touch ? 1 : 0;
  int global_touch_int = local_touch_int;
  const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
  collectives.allreduce_sum(local_touch_int, global_touch_int);
  return global_touch_int != 0;
}

bool grouped_bordered_touch_constraint_block(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    int mom_ncomp,
    int con_ncomp)
{
  bool local_touch = false;
  for (const auto& group : lhs.grouped_bordered_field_couplings) {
    if (!group.active) {
      continue;
    }
    for (const auto& mode : group.modes) {
      const auto& left_entries = !mode.left_scaled.empty() ? mode.left_scaled : mode.left;
      const auto& right_entries = !mode.right_scaled.empty() ? mode.right_scaled : mode.right;
      if (entries_touch_constraint_block(left_entries, mom_ncomp, con_ncomp) ||
          entries_touch_constraint_block(right_entries, mom_ncomp, con_ncomp)) {
        local_touch = true;
        break;
      }
    }
    if (local_touch) {
      break;
    }
  }
  int local_touch_int = local_touch ? 1 : 0;
  int global_touch_int = local_touch_int;
  const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
  collectives.allreduce_sum(local_touch_int, global_touch_int);
  return global_touch_int != 0;
}

double sparse_overlap_dot_owned_local(
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& left_entries,
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& right_entries)
{
  if (left_entries.empty() || right_entries.empty()) {
    return 0.0;
  }

  struct EntryKey {
    fsils_int node = -1;
    int full_component = -1;
    bool operator==(const EntryKey& other) const noexcept
    {
      return node == other.node && full_component == other.full_component;
    }
  };
  struct EntryKeyHash {
    std::size_t operator()(const EntryKey& key) const noexcept
    {
      std::size_t seed = std::hash<fsils_int>{}(key.node);
      seed ^= std::hash<int>{}(key.full_component) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
      return seed;
    }
  };

  std::unordered_map<EntryKey, double, EntryKeyHash> left_values;
  left_values.reserve(left_entries.size());
  for (const auto& entry : left_entries) {
    if (entry.node < 0 || entry.full_component < 0 || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    left_values[{entry.node, entry.full_component}] += entry.value;
  }

  double dot = 0.0;
  for (const auto& entry : right_entries) {
    if (entry.node < 0 || entry.full_component < 0 || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    const auto it = left_values.find({entry.node, entry.full_component});
    if (it != left_values.end()) {
      dot += entry.value * it->second;
    }
  }
  return dot;
}

bool sparse_entry_sets_match(
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& left_entries,
    const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& right_entries,
    double tol = 1e-30)
{
  if (left_entries.size() != right_entries.size()) {
    return false;
  }
  for (std::size_t i = 0; i < left_entries.size(); ++i) {
    const auto& left = left_entries[i];
    const auto& right = right_entries[i];
    if (left.node != right.node || left.full_component != right.full_component ||
        std::abs(left.value - right.value) > tol) {
      return false;
    }
  }
  return true;
}

bool reduced_update_has_distinct_left_right(
    const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update)
{
  const auto& left_entries =
      !update.left_scaled_owned.empty() ? update.left_scaled_owned
      : !update.left_owned.empty()      ? update.left_owned
      : !update.left_scaled.empty()     ? update.left_scaled
                                        : update.left;
  const auto& right_entries =
      !update.right_scaled_owned.empty() ? update.right_scaled_owned
      : !update.right_owned.empty()      ? update.right_owned
      : !update.right_scaled.empty()     ? update.right_scaled
                                         : update.right;
  return !sparse_entry_sets_match(left_entries, right_entries);
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
  const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
  collectives.allreduce_sum(values);
}

void dense_axpy(int dof,
                fsils_int nNo,
                const Array<double>& src,
                double scale,
                Array<double>& dst)
{
  if (std::abs(scale) <= 1e-30 || src.nrows() != dof || src.ncols() != nNo ||
      dst.nrows() != dof || dst.ncols() != nNo) {
    return;
  }

  #pragma omp parallel for schedule(static)
  for (fsils_int node = 0; node < nNo; ++node) {
    for (int comp = 0; comp < dof; ++comp) {
      dst(comp, node) += scale * src(comp, node);
    }
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

void build_explicit_schur_corrections(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                      int mom_start,
                                      int mom_ncomp,
                                      int con_start,
                                      int con_ncomp,
                                      SchurPreconditionerData& pc)
{
  pc.explicit_low_rank_correction =
      fe_fsi_linear_solver::distributed_low_rank_correction::build(
          lhs, mom_start, mom_ncomp, con_start, con_ncomp);
}

void apply_explicit_constraint_schur_corrections(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                                 const SchurPreconditionerData& pc,
                                                 const Array<double>& in_constraint,
                                                 Array<double>& gp,
                                                 Array<double>& sp)
{
  fe_fsi_linear_solver::distributed_low_rank_correction::apply_constraint_driven(
      lhs, pc.explicit_low_rank_correction, in_constraint, gp, sp);
}

void apply_explicit_momentum_schur_corrections(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                               const SchurPreconditionerData& pc,
                                               const Array<double>& in_momentum,
                                               Array<double>& dgp)
{
  fe_fsi_linear_solver::distributed_low_rank_correction::apply_momentum_driven(
      lhs, pc.explicit_low_rank_correction, in_momentum, dgp);
}

void build_grouped_momentum_hat_low_rank_correction(
    const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr,
    const Vector<fsils_int>& diagPtr,
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    int mom_ncomp,
    fsils_int nNo,
    MomentumHatData& hat)
{
  hat.low_rank_right.clear();
  hat.low_rank_preconditioned_left.clear();
  hat.low_rank_left.clear();
  hat.low_rank_preconditioned_right_t.clear();
  hat.low_rank_inner_inv.clear();
  hat.low_rank_inner_inv_t.clear();

  struct GroupModes {
    const fe_fsi_linear_solver::FSILS_groupedBorderedFieldCouplingType* group{nullptr};
    std::vector<Array<double>> left_modes;
    std::vector<Array<double>> right_modes;
  };

  std::vector<GroupModes> groups;
  groups.reserve(lhs.grouped_bordered_field_couplings.size());
  int total_rank = 0;

  for (const auto& group : lhs.grouped_bordered_field_couplings) {
    if (!group.active || group.modes.empty()) {
      continue;
    }

    GroupModes kept;
    kept.group = &group;
    kept.left_modes.reserve(group.modes.size());
    kept.right_modes.reserve(group.modes.size());

    for (const auto& mode : group.modes) {
      const auto& left_entries =
          !mode.left_scaled_owned.empty() ? mode.left_scaled_owned
          : !mode.left_owned.empty()      ? mode.left_owned
          : !mode.left_scaled.empty()     ? mode.left_scaled
                                          : mode.left;
      const auto& right_entries =
          !mode.right_scaled_owned.empty() ? mode.right_scaled_owned
          : !mode.right_owned.empty()      ? mode.right_owned
          : !mode.right_scaled.empty()     ? mode.right_scaled
                                           : mode.right;

      Array<double> left_mode;
      Array<double> right_mode;
      fill_projected_reduced_vector(lhs, mode, left_entries, mom_ncomp, left_mode);
      fill_projected_reduced_vector(lhs, mode, right_entries, mom_ncomp, right_mode);

      kept.left_modes.push_back(std::move(left_mode));
      kept.right_modes.push_back(std::move(right_mode));
    }

    if (!kept.left_modes.empty()) {
      total_rank += static_cast<int>(kept.left_modes.size());
      groups.push_back(std::move(kept));
    }
  }

  if (total_rank <= 0) {
    return;
  }

  std::vector<double> dense_c_inv(static_cast<std::size_t>(total_rank) *
                                  static_cast<std::size_t>(total_rank),
                                  0.0);
  hat.low_rank_left.reserve(static_cast<std::size_t>(total_rank));
  hat.low_rank_right.reserve(static_cast<std::size_t>(total_rank));

  int offset = 0;
  for (const auto& kept : groups) {
    const auto& group = *kept.group;
    const int rank = static_cast<int>(kept.left_modes.size());
    for (int i = 0; i < rank; ++i) {
      hat.low_rank_left.push_back(kept.left_modes[static_cast<std::size_t>(i)]);
      hat.low_rank_right.push_back(kept.right_modes[static_cast<std::size_t>(i)]);
      for (int j = 0; j < rank; ++j) {
        dense_c_inv[static_cast<std::size_t>(offset + i) * static_cast<std::size_t>(total_rank) +
                    static_cast<std::size_t>(offset + j)] =
            -group.aux_matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                              static_cast<std::size_t>(j)];
      }
    }
    offset += rank;
  }

  hat.low_rank_preconditioned_left = hat.low_rank_left;
  hat.low_rank_preconditioned_right_t = hat.low_rank_right;
  for (auto& vec : hat.low_rank_preconditioned_left) {
    apply_momentum_hat(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat, vec);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    halo.sync_vector(mom_ncomp, vec, skip_post_solve_overlap_sum());
  }
  for (auto& vec : hat.low_rank_preconditioned_right_t) {
    apply_momentum_hat_transpose(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat, vec);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    halo.sync_vector(mom_ncomp, vec, skip_post_solve_overlap_sum());
  }

  std::vector<double> dense_m = dense_c_inv;
  std::vector<double> dense_mt(static_cast<std::size_t>(total_rank) *
                                   static_cast<std::size_t>(total_rank),
                               0.0);
  for (int i = 0; i < total_rank; ++i) {
    for (int j = 0; j < total_rank; ++j) {
      dense_mt[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
               static_cast<std::size_t>(j)] =
          dense_c_inv[static_cast<std::size_t>(j) * static_cast<std::size_t>(total_rank) +
                      static_cast<std::size_t>(i)];
    }
  }

  for (int i = 0; i < total_rank; ++i) {
    for (int j = 0; j < total_rank; ++j) {
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
              static_cast<std::size_t>(j)] +=
          dot::fsils_dot_v(
              mom_ncomp,
              lhs.mynNo,
              const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
              hat.low_rank_right[static_cast<std::size_t>(i)],
              hat.low_rank_preconditioned_left[static_cast<std::size_t>(j)]);
      dense_mt[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
               static_cast<std::size_t>(j)] +=
          dot::fsils_dot_v(
              mom_ncomp,
              lhs.mynNo,
              const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
              hat.low_rank_left[static_cast<std::size_t>(i)],
              hat.low_rank_preconditioned_right_t[static_cast<std::size_t>(j)]);
    }
  }

  hat.low_rank_inner_inv.assign(dense_m.size(), 0.0);
  hat.low_rank_inner_inv_t.assign(dense_mt.size(), 0.0);
  if (!invert_dense_block(total_rank, dense_m.data(), hat.low_rank_inner_inv.data()) ||
      !invert_dense_block(total_rank, dense_mt.data(), hat.low_rank_inner_inv_t.data())) {
    hat.low_rank_right.clear();
    hat.low_rank_preconditioned_left.clear();
    hat.low_rank_left.clear();
    hat.low_rank_preconditioned_right_t.clear();
    hat.low_rank_inner_inv.clear();
    hat.low_rank_inner_inv_t.clear();
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
  const auto momentum_operator =
      dso::SparseOperatorBundle(lhs, rowPtr, colPtr).vector(mom_ncomp, K_eff);
  const int block_entries = mom_ncomp * mom_ncomp;
  const int system_dof = (lhs.system_dof > 0) ? lhs.system_dof : mom_ncomp;
  const bool has_grouped_bordered = !lhs.grouped_bordered_field_couplings.empty();

  using NodeComponentEntries = std::unordered_map<fsils_int, std::vector<std::pair<int, double>>>;

  auto component_index = [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
                             int full_comp) -> int {
    return fe_fsi_linear_solver::fsils_reduced_local_component(update,
                                                               full_comp,
                                                               mom_ncomp,
                                                               system_dof);
  };

  auto collect_owned_by_node =
      [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
          const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& entries,
          NodeComponentEntries& by_node) {
    by_node.clear();
    for (const auto& entry : entries) {
      if (entry.node < 0 || std::abs(entry.value) <= 1e-30) {
        continue;
      }
      const fsils_int node = entry.node;
      const int local_comp = component_index(update, entry.full_component);
      if (node < 0 || node >= nNo || local_comp < 0 || local_comp >= mom_ncomp) {
        continue;
      }
      by_node[node].emplace_back(local_comp, entry.value);
    }
  };

  auto add_graph_outer_product = [&](const NodeComponentEntries& left_by_node,
                                     const NodeComponentEntries& right_by_node,
                                     double coeff) {
    if (std::abs(coeff) <= 1e-30 || left_by_node.empty() || right_by_node.empty()) {
      return;
    }

    for (const auto& [row_node, left_vals] : left_by_node) {
      for (const auto& [col_node, right_vals] : right_by_node) {
        const fsils_int nz = find_col_in_row(rowPtr, colPtr, row_node, col_node);
        if (nz < 0) {
          continue;
        }
        double* block = block_ptr(K_eff, block_entries, nz);
        for (const auto& [li, lval] : left_vals) {
          for (const auto& [rj, rval] : right_vals) {
            block[li * mom_ncomp + rj] += coeff * lval * rval;
          }
        }
      }
    }
  };

  auto add_graph_update_entries =
      [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update) {
    if (!update.active || std::abs(update.sigma) <= 1e-30) {
      return;
    }
    if (update.grouped_coupling_id == kNativeFaceDuplicateCouplingId) {
      return;
    }
    if (has_grouped_bordered && update.grouped_coupling_id >= 0) {
      return;
    }

    const auto& left_entries =
        !update.left_scaled_owned.empty() ? update.left_scaled_owned : update.left_owned;
    const auto& right_entries =
        !update.right_scaled_owned.empty() ? update.right_scaled_owned : update.right_owned;
    NodeComponentEntries left_by_node;
    NodeComponentEntries right_by_node;
    collect_owned_by_node(update, left_entries, left_by_node);
    collect_owned_by_node(update, right_entries, right_by_node);
    add_graph_outer_product(left_by_node, right_by_node, update.sigma);
  };

  auto add_graph_face_entries =
      [&](const fe_fsi_linear_solver::FSILS_faceType& face) {
    if (!face.coupledFlag || std::abs(face.res) <= 1e-30 || face.nNo <= 0) {
      return;
    }

    NodeComponentEntries face_by_node;
    const int face_dof = std::min(face.dof, mom_ncomp);
    for (int a = 0; a < face.nNo; ++a) {
      const fsils_int node = face.glob(a);
      if (node < 0 || node >= nNo) {
        continue;
      }
      for (int comp = 0; comp < face_dof; ++comp) {
        const double value = face.valM(comp, a);
        if (std::abs(value) <= 1e-30) {
          continue;
        }
        face_by_node[node].emplace_back(comp, value);
      }
    }

    add_graph_outer_product(face_by_node, face_by_node, face.res);
  };

  for (const auto& update : lhs.reduced_updates) {
    add_graph_update_entries(update);
  }

  auto build_candidate =
      [&](SchurMomentumApproximationType candidate_approx) -> MomentumHatData {
    MomentumHatData candidate{};
    candidate.mynNo = lhs.mynNo;
    candidate.commu = &lhs.commu;
    switch (candidate_approx) {
      case SchurMomentumApproximationType::DIAG_K:
        build_point_inverse_blocks(diagPtr, mom_ncomp, nNo, K_eff, /*diagonal_only=*/true, candidate.point_inv);
        build_sparse_block_diagonal_inverse(diagPtr, mom_ncomp, nNo, K_eff.ncols(), candidate.point_inv,
                                            candidate.sparse_inverse);
        build_grouped_momentum_hat_low_rank_correction(rowPtr, colPtr, diagPtr, lhs,
                                                       mom_ncomp, nNo, candidate);
        return candidate;
      case SchurMomentumApproximationType::BLOCKDIAG_K:
        build_point_inverse_blocks(diagPtr, mom_ncomp, nNo, K_eff, /*diagonal_only=*/false, candidate.point_inv);
        build_sparse_block_diagonal_inverse(diagPtr, mom_ncomp, nNo, K_eff.ncols(), candidate.point_inv,
                                            candidate.sparse_inverse);
        build_grouped_momentum_hat_low_rank_correction(rowPtr, colPtr, diagPtr, lhs,
                                                       mom_ncomp, nNo, candidate);
        return candidate;
      case SchurMomentumApproximationType::ILU_K:
      {
        factorize_block_ilu0(rowPtr, colPtr, diagPtr, nNo, mom_ncomp, K_eff,
                             candidate.ilu_factors, candidate.ilu_diag_inv);
        candidate.point_inv = candidate.ilu_diag_inv;
        build_sparse_block_ilu_inverse(rowPtr, colPtr, diagPtr, nNo, K_eff.ncols(), mom_ncomp,
                                       candidate.ilu_factors, candidate.ilu_diag_inv, candidate.sparse_inverse);
        prune_sparse_momentum_hat(rowPtr, colPtr, diagPtr, nNo, mom_ncomp, candidate.sparse_inverse);
        candidate.use_ilu = true;
        build_grouped_momentum_hat_low_rank_correction(rowPtr, colPtr, diagPtr, lhs,
                                                       mom_ncomp, nNo, candidate);
        return candidate;
      }
      case SchurMomentumApproximationType::ASM_K: {
        build_zero_overlap_asm_inverse(rowPtr, colPtr, nNo, K_eff.ncols(), mom_ncomp, K_eff,
                                       candidate.asm_patch_ptr, candidate.asm_patch_nodes,
                                       candidate.asm_patch_matrix_ptr, candidate.asm_patch_inverse,
                                       candidate.sparse_inverse);
        build_point_inverse_blocks(diagPtr, mom_ncomp, nNo, K_eff, /*diagonal_only=*/false, candidate.point_inv);
        candidate.use_asm = true;
        build_grouped_momentum_hat_low_rank_correction(rowPtr, colPtr, diagPtr, lhs,
                                                       mom_ncomp, nNo, candidate);
        return candidate;
      }
    }
    return candidate;
  };

  auto score_candidate = [&](const MomentumHatData& candidate) -> double {
    long double numerator = 0.0L;
    long double denominator = 0.0L;
    Array<double> probe(mom_ncomp, nNo);
    Array<double> approx_apply(mom_ncomp, nNo);
    Array<double> residual(mom_ncomp, nNo);
    Array<double> probe_t(mom_ncomp, nNo);
    Array<double> approx_apply_t(mom_ncomp, nNo);
    Array<double> residual_t(mom_ncomp, nNo);

    for (const auto& group : lhs.grouped_bordered_field_couplings) {
      if (!group.active) {
        continue;
      }
      for (const auto& mode : group.modes) {
        const auto& left_entries =
            !mode.left_scaled_owned.empty() ? mode.left_scaled_owned
            : !mode.left_owned.empty()      ? mode.left_owned
            : !mode.left_scaled.empty()     ? mode.left_scaled
                                            : mode.left;
        fill_projected_reduced_vector(lhs, mode, left_entries, mom_ncomp, probe);
        approx_apply = probe;
        apply_momentum_hat(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, candidate, approx_apply);
        const fe_fsi_linear_solver::HaloExchange halo(lhs);
        halo.sync_vector(mom_ncomp, approx_apply, skip_post_solve_overlap_sum());
        momentum_operator.apply(
            dso::ghost_synced_input(mom_ncomp, approx_apply),
            dso::ghost_synced_output(mom_ncomp, residual));
        omp_la::omp_axpby_v(mom_ncomp, nNo, residual, residual, -1.0, probe);
        const double den = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                            const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                            probe, probe);
        const double num = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                            const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                            residual, residual);
        denominator += static_cast<long double>(den);
        numerator += static_cast<long double>(num);

        const auto& right_entries =
            !mode.right_scaled_owned.empty() ? mode.right_scaled_owned
            : !mode.right_owned.empty()      ? mode.right_owned
            : !mode.right_scaled.empty()     ? mode.right_scaled
                                             : mode.right;
        fill_projected_reduced_vector(lhs, mode, right_entries, mom_ncomp, probe_t);
        approx_apply_t = probe_t;
        apply_momentum_hat_transpose(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, candidate, approx_apply_t);
        halo.sync_vector(mom_ncomp, approx_apply_t, skip_post_solve_overlap_sum());
        multiply_rect_transpose_local(rowPtr, colPtr, nNo,
                                      mom_ncomp, mom_ncomp, K_eff,
                                      approx_apply_t, residual_t);
        omp_la::omp_axpby_v(mom_ncomp, nNo, residual_t, residual_t, -1.0, probe_t);
        const double den_t = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                              const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                              probe_t, probe_t);
        const double num_t = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                              const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                              residual_t, residual_t);
        denominator += static_cast<long double>(den_t);
        numerator += static_cast<long double>(num_t);
      }
    }

    if (!(denominator > 1e-30L)) {
      return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(numerator / denominator);
  };

  if (has_grouped_bordered && approx == SchurMomentumApproximationType::ILU_K) {
    MomentumHatData ilu_hat = build_candidate(SchurMomentumApproximationType::ILU_K);
    MomentumHatData asm_hat = build_candidate(SchurMomentumApproximationType::ASM_K);
    const double ilu_score = score_candidate(ilu_hat);
    const double asm_score = score_candidate(asm_hat);
    if (std::isfinite(asm_score) && asm_score < ilu_score) {
      return asm_hat;
    }
    return ilu_hat;
  }

  return build_candidate(approx);
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
  const auto d_operator =
      dso::SparseOperatorBundle(lhs, lhs.rowPtr, lhs.colPtr)
          .rectangular(con_ncomp, mom_ncomp, D);
  struct ReducedColumn {
    Array<double> momentum_right_owned;
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

    const auto& left_entries =
        !update.left_scaled_owned.empty() ? update.left_scaled_owned
        : !update.left_owned.empty()      ? update.left_owned
        : !update.left_scaled.empty()     ? update.left_scaled
                                          : update.left;
    const auto& right_entries =
        !update.right_scaled_owned.empty() ? update.right_scaled_owned
        : !update.right_owned.empty()      ? update.right_owned
        : !update.right_scaled.empty()     ? update.right_scaled
                                           : update.right;

    ReducedColumn col;
    fill_projected_reduced_vector(lhs, update, left_entries, mom_ncomp, col.momentum_left_hat);
    fill_projected_reduced_vector(lhs, update, right_entries, mom_ncomp, col.momentum_right_owned);

    Array<double> momentum_right_t;
    fill_projected_reduced_vector(lhs, update, right_entries, mom_ncomp, momentum_right_t);

    apply_momentum_hat(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                       mom_ncomp, lhs.nNo, momentum_hat, col.momentum_left_hat);
    apply_momentum_hat_transpose(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                 mom_ncomp, lhs.nNo, momentum_hat, momentum_right_t);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    halo.sync_vector(mom_ncomp, col.momentum_left_hat, skip_post_solve_overlap_sum());
    halo.sync_vector(mom_ncomp, momentum_right_t, skip_post_solve_overlap_sum());

    col.schur_left.resize(con_ncomp, lhs.nNo);
    d_operator.apply(
        dso::ghost_synced_input(mom_ncomp, col.momentum_left_hat),
        dso::ghost_synced_output(con_ncomp, col.schur_left));
    multiply_rect_transpose_local(lhs.rowPtr, lhs.colPtr, lhs.nNo,
                                  mom_ncomp, con_ncomp, G,
                                  momentum_right_t, col.schur_right);
    columns.push_back(std::move(col));
  };

  auto append_face_column = [&](const fe_fsi_linear_solver::FSILS_faceType& face) {
    if (!face.coupledFlag || std::abs(face.res) <= 1e-30) {
      return;
    }

    ReducedColumn col;
    fill_projected_face_vector(lhs, face, mom_ncomp, col.momentum_left_hat);
    col.momentum_right_owned = col.momentum_left_hat;
    Array<double> momentum_right_t = col.momentum_left_hat;

    apply_momentum_hat(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                       mom_ncomp, lhs.nNo, momentum_hat, col.momentum_left_hat);
    apply_momentum_hat_transpose(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                 mom_ncomp, lhs.nNo, momentum_hat, momentum_right_t);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    halo.sync_vector(mom_ncomp, col.momentum_left_hat, skip_post_solve_overlap_sum());
    halo.sync_vector(mom_ncomp, momentum_right_t, skip_post_solve_overlap_sum());

    col.schur_left.resize(con_ncomp, lhs.nNo);
    d_operator.apply(
        dso::ghost_synced_input(mom_ncomp, col.momentum_left_hat),
        dso::ghost_synced_output(con_ncomp, col.schur_left));
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
    if (update.grouped_coupling_id == kNativeFaceDuplicateCouplingId) {
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
                            std::vector<double>{safe_inverse(update.sigma)},
                            /*block_rank=*/1);
  }

  for (const auto& face : lhs.face) {
    const int old_rank = static_cast<int>(columns.size());
    append_face_column(face);
    if (static_cast<int>(columns.size()) != old_rank + 1) {
      continue;
    }
    append_dense_seed_block(dense_seed, old_rank,
                            std::vector<double>{safe_inverse(face.res)},
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
    for (int j = 0; j < rank; ++j) {
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(j)] +=
          dense_dense_owned_dot_local(lhs, mom_ncomp,
                                      columns[static_cast<std::size_t>(i)].momentum_right_owned,
                                      columns[static_cast<std::size_t>(j)].momentum_left_hat);
    }
  }
  allreduce_sum_in_place(lhs, dense_m);

  pc.low_rank_right.clear();
  pc.low_rank_preconditioned_left.clear();
  pc.low_rank_right.reserve(columns.size());
  pc.low_rank_preconditioned_left.reserve(columns.size());
  for (std::size_t column_index = 0; column_index < columns.size(); ++column_index) {
    auto& column = columns[column_index];
    pc.low_rank_right.push_back(column.schur_right);
    Array<double> z = column.schur_left;
    apply_base_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                    pc, con_ncomp, lhs.nNo, z);
    sync_schur_overlap(lhs, con_ncomp, z);
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
  const bool has_face_corrections = std::any_of(lhs.face.begin(), lhs.face.end(),
      [](const fe_fsi_linear_solver::FSILS_faceType& face) {
        return face.coupledFlag && std::abs(face.res) > 1e-30;
      });
  const bool has_exact_schur_corrections =
      !lhs.reduced_updates.empty() ||
      !lhs.grouped_bordered_field_couplings.empty() ||
      has_face_corrections;
  const bool need_momentum_hat =
      has_exact_schur_corrections ||
      (preconditioner == SchurPreconditionerType::ALGEBRAIC_SHAT);
  build_point_inverse_blocks(lhs.diagPtr, con_ncomp, lhs.nNo, L, diagonal_only, pc.point_inv);
  if (!lhs.reduced_updates.empty() || !lhs.grouped_bordered_field_couplings.empty()) {
    // The OOP BlockSchur path permutes the per-node ordering to the explicit
    // momentum/constraint block layout before entering FSILS. The Schur
    // subsolver does not carry the original block starts, so project the exact
    // outlet corrections using the FSILS-local block order.
    build_explicit_schur_corrections(lhs, /*mom_start=*/0, mom_ncomp,
                                     /*con_start=*/mom_ncomp, con_ncomp, pc);
  }

  if (need_momentum_hat) {
    pc.momentum_hat =
        build_momentum_hat_data(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, lhs.nNo,
                                mom_ncomp, K, ls.schur_momentum_approximation);
  }

  if (preconditioner == SchurPreconditionerType::ILU_L) {
    factorize_block_ilu0(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, con_ncomp, L,
                         pc.ilu_factors, pc.ilu_diag_inv);
    pc.use_ilu = true;
    if (has_exact_schur_corrections) {
      build_reduced_schur_correction(lhs, mom_ncomp, con_ncomp, D, G, pc.momentum_hat, pc);
    }
    return pc;
  }

  if (preconditioner == SchurPreconditionerType::ALGEBRAIC_SHAT) {
    const auto& cfg = schur_sparsity_control();
    Array<double> shat(L.nrows(), L.ncols());
    assemble_algebraic_schur(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, mom_ncomp, con_ncomp,
                             D, G, L, pc.momentum_hat.sparse_inverse, shat);
    factorize_block_ilu0(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, con_ncomp, shat,
                         pc.ilu_factors, pc.ilu_diag_inv);
    pc.use_ilu = true;
    pc.operator_D = &D;
    pc.operator_G = &G;
    pc.operator_L_storage = L;
    pc.operator_L = &pc.operator_L_storage;
    pc.operator_mom_ncomp = mom_ncomp;
    pc.operator_con_ncomp = con_ncomp;
    pc.operator_refinement_steps = cfg.operator_refinement_steps;
    pc.operator_refinement_omega = cfg.operator_refinement_omega;
  }

  if (has_exact_schur_corrections) {
    build_reduced_schur_correction(lhs, mom_ncomp, con_ncomp, D, G, pc.momentum_hat, pc);
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
  const auto schur_ops = dsb::make_multi_constraint_schur_system(
      lhs, mom_ncomp, con_ncomp, *pc.operator_D, *pc.operator_G, *pc.operator_L);

  pc.scratch_gp.resize(mom_ncomp, nNo);
  pc.scratch_hgp.resize(mom_ncomp, nNo);
  pc.scratch_sp.resize(con_ncomp, nNo);
  pc.scratch_dgp.resize(con_ncomp, nNo);

  schur_ops.GL.apply(
      dso::ghost_synced_input(con_ncomp, in_vec),
      dso::ghost_synced_output(mom_ncomp, pc.scratch_gp),
      dso::ghost_synced_output(con_ncomp, pc.scratch_sp));
  apply_explicit_constraint_schur_corrections(lhs, pc, in_vec, pc.scratch_gp, pc.scratch_sp);

  pc.scratch_hgp = pc.scratch_gp;
  apply_momentum_hat(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, pc.momentum_hat, pc.scratch_hgp);
  const fe_fsi_linear_solver::HaloExchange hat_halo(lhs);
  hat_halo.sync_vector(mom_ncomp, pc.scratch_hgp, skip_post_solve_overlap_sum());

  schur_ops.D.apply(
      dso::ghost_synced_input(mom_ncomp, pc.scratch_hgp),
      dso::ghost_synced_output(con_ncomp, pc.scratch_dgp));
  apply_explicit_momentum_schur_corrections(lhs, pc, pc.scratch_hgp, pc.scratch_dgp);

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
  sync_schur_overlap(lhs, con_ncomp, x);
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
    sync_schur_overlap(lhs, con_ncomp, pc.scratch_correction);
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

void schur_face_only_legacy(const dsb::ScalarBlockSchurSystem& system,
                            fe_fsi_linear_solver::FSILS_subLsType& ls,
                            Vector<double>& R)
{
  using namespace fe_fsi_linear_solver;
  auto& lhs = *system.lhs;
  const int nsd = system.momentum_components;
  const auto& D = system.D;
  const auto& L_values = *system.L_values;
  const auto& GL = system.GL;

  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;

  ls.ws.ensure_bicgs_s(nNo);
  auto& P = ls.ws.bicgs_Ps;
  auto& Rh = ls.ws.bicgs_Rhs;
  auto& X = ls.ws.bicgs_Xs;
  auto& V = ls.ws.bicgs_Vs;
  auto& S = ls.ws.bicgs_Ss;
  auto& T = ls.ws.bicgs_Ts;

  Array<double> GP(nsd, nNo);
  Vector<double> SP(nNo), DGP(nNo);
  Vector<double> M_inv(nNo);
  Vector<double> in_sync(nNo);
  for (fsils_int i = 0; i < nNo; ++i) {
    const double diag_val = L_values(lhs.diagPtr(i));
    M_inv(i) = (std::abs(diag_val) > 1e-12) ? 1.0 / diag_val : 1.0;
  }

  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; ++i) {
    R(i) *= M_inv(i);
  }

  ls.callD = fsils_cpu_t();
  ls.suc = false;

  double err = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  const double err_initial = err;
  double errO = err;
  ls.iNorm = err;
  const double eps = std::max(ls.absTol, ls.relTol * err);
  double rho = err * err;
  double beta = rho;

  X = 0.0;
  P = R;
  Rh = R;
  int i_itr = 1;

  auto apply_schur_operator = [&](const Vector<double>& in_vec, Vector<double>& out_vec) {
    const Vector<double>* op_in = &in_vec;
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    if (halo.has_overlap() && !skip_post_solve_overlap_sum()) {
      // BiCGStab updates overlap vectors locally; synchronize before reuse in the
      // next Schur matvec so owned values drive ghost entries consistently.
      in_sync = in_vec;
      halo.sync_scalar(in_sync);
      op_in = &in_sync;
    }

    GL.apply(
        dso::ghost_synced_input(*op_in),
        dso::ghost_synced_output(nsd, GP),
        dso::ghost_synced_output(SP));
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, nsd, GP, GP);
    if (halo.has_overlap() && !skip_post_solve_overlap_sum()) {
      // The coupled-face correction modifies a momentum overlap field in place.
      // Synchronize before reusing it in D*(...) on neighboring owned rows.
      halo.sync_vector(nsd, GP);
    }
    D.apply(
        dso::ghost_synced_input(nsd, GP),
        dso::ghost_synced_output(DGP));

    #pragma omp parallel for schedule(static)
    for (fsils_int i = 0; i < nNo; ++i) {
      out_vec(i) = M_inv(i) * (SP(i) - DGP(i));
    }

    halo.sync_scalar(out_vec, skip_post_solve_overlap_sum());
  };

  int active_coupled_faces = static_cast<int>(std::count_if(
      lhs.face.begin(), lhs.face.end(),
      [](const auto& face) {
        return face.coupledFlag && std::abs(face.res) > 1e-30 && face.nNo > 0;
      }));
  const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
  {
    int global_active_coupled_faces = active_coupled_faces;
    collectives.allreduce_sum(active_coupled_faces, global_active_coupled_faces);
    active_coupled_faces = global_active_coupled_faces;
  }
  const bool use_multi_face_gmres =
      collectives.distributed() &&
      active_coupled_faces > 1 &&
      (std::getenv("SVMP_FSILS_DISABLE_MULTI_FACE_LEGACY_GMRES") == nullptr);

  if (use_multi_face_gmres) {
    const int restart_dim =
        std::max(1, std::min((ls.sD > 0 ? ls.sD : ls.mItr), ls.mItr));
    ls.ws.ensure_gmres_s(nNo, restart_dim);
    auto& h = ls.ws.h;
    auto& u = ls.ws.u2;
    auto& Xg = ls.ws.Xs;
    auto& y = ls.ws.y;
    auto& c = ls.ws.c;
    auto& s = ls.ws.s;
    auto& errv = ls.ws.err;

    Vector<double> residual(nNo), ax(nNo);
    ls.callD = fsils_cpu_t();
    ls.suc = false;
    ls.itr = 0;
    Xg = 0.0;

    const double rhs_norm = norm::fsi_ls_norms(mynNo, lhs.commu, R);
    ls.iNorm = rhs_norm;
    const double gmres_eps = std::max(ls.absTol, ls.relTol * rhs_norm);
    int restart_cycles = 0;
    int total_itr = 0;

    if (rhs_norm <= ls.absTol) {
      R = Xg;
      ls.fNorm = rhs_norm;
      ls.callD = fsils_cpu_t() - ls.callD;
      ls.dB = 0.0;
      ls.suc = true;
      ls.stats.record_call(ls.itr - itr_before,
                           restart_cycles,
                           fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                           /*setup_seconds=*/0.0,
                           ls.callD - callD_before);
      return;
    }

    while (total_itr < ls.mItr) {
      ++restart_cycles;

      apply_schur_operator(Xg, ax);
      omp_la::omp_axpby_s(nNo, residual, R, -1.0, ax);
      const double beta_gmres = norm::fsi_ls_norms(mynNo, lhs.commu, residual);
      if (beta_gmres < gmres_eps) {
        ls.suc = true;
        ls.fNorm = beta_gmres;
        break;
      }

      h = 0.0;
      for (int i = 0; i <= restart_dim; ++i) {
        errv(i) = 0.0;
        if (i < restart_dim) {
          c(i) = 0.0;
          s(i) = 0.0;
          y(i) = 0.0;
        }
      }
      errv(0) = beta_gmres;

      auto u0 = u.rcol(0);
      u0 = residual;
      omp_la::omp_mul_s(nNo, 1.0 / beta_gmres, u0);

      int last_i = -1;
      bool terminated_inner = false;
      for (int i = 0; i < restart_dim && total_itr < ls.mItr; ++i) {
        last_i = i;
        auto ui = u.rcol(i);
        auto uip1 = u.rcol(i + 1);
        apply_schur_operator(ui, uip1);

        for (int j = 0; j <= i; ++j) {
          h(j, i) = dot::fsils_dot_s(mynNo, lhs.commu, u.rcol(j), uip1);
          omp_la::omp_sum_s(nNo, -h(j, i), uip1, u.rcol(j));
        }

        h(i + 1, i) = norm::fsi_ls_norms(mynNo, lhs.commu, uip1);
        const bool breakdown =
            !(h(i + 1, i) >
              std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0));
        if (!breakdown) {
          omp_la::omp_mul_s(nNo, 1.0 / h(i + 1, i), uip1);
        } else {
          h(i + 1, i) = 0.0;
        }

        for (int j = 0; j < i; ++j) {
          const double tmp_h = c(j) * h(j, i) + s(j) * h(j + 1, i);
          h(j + 1, i) = -s(j) * h(j, i) + c(j) * h(j + 1, i);
          h(j, i) = tmp_h;
        }

        const double hypot_h = std::hypot(h(i, i), h(i + 1, i));
        if (hypot_h > std::numeric_limits<double>::epsilon()) {
          c(i) = h(i, i) / hypot_h;
          s(i) = h(i + 1, i) / hypot_h;
        } else {
          c(i) = 1.0;
          s(i) = 0.0;
        }

        h(i, i) = hypot_h;
        h(i + 1, i) = 0.0;
        errv(i + 1) = -s(i) * errv(i);
        errv(i) = c(i) * errv(i);

        ++total_itr;
        ls.itr = total_itr;

        if (std::abs(errv(i + 1)) < gmres_eps || breakdown) {
          terminated_inner = true;
          break;
        }
      }

      if (last_i < 0) {
        break;
      }

      for (int i = 0; i <= last_i; ++i) {
        y(i) = errv(i);
      }
      for (int j = last_i; j >= 0; --j) {
        for (int k = j + 1; k <= last_i; ++k) {
          y(j) -= h(j, k) * y(k);
        }
        if (std::abs(h(j, j)) > std::numeric_limits<double>::epsilon()) {
          y(j) /= h(j, j);
        } else {
          y(j) = 0.0;
        }
      }

      for (int j = 0; j <= last_i; ++j) {
        if (std::abs(y(j)) > 1e-30) {
          omp_la::omp_sum_s(nNo, y(j), Xg, u.rcol(j));
        }
      }

      ls.fNorm = std::abs(errv(last_i + 1));
      if (ls.fNorm < gmres_eps) {
        ls.suc = true;
        break;
      }

      if (!terminated_inner) {
        continue;
      }
    }

    if (!ls.suc) {
      apply_schur_operator(Xg, ax);
      omp_la::omp_axpby_s(nNo, residual, R, -1.0, ax);
      ls.fNorm = norm::fsi_ls_norms(mynNo, lhs.commu, residual);
      ls.suc = (ls.fNorm < gmres_eps);
    }

    R = Xg;
    ls.callD = fsils_cpu_t() - ls.callD;
    ls.dB = (rhs_norm < std::numeric_limits<double>::epsilon() || ls.fNorm <= 0.0)
                ? 0.0
                : 10.0 * std::log(ls.fNorm / rhs_norm);
    ls.stats.record_call(ls.itr - itr_before,
                         restart_cycles,
                         fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                         /*setup_seconds=*/0.0,
                         ls.callD - callD_before);
    return;
  }

  for (int i = 0; i < ls.mItr; ++i) {
    if (err < eps) {
      ls.suc = true;
      break;
    }

    apply_schur_operator(P, V);

    const double denom_alpha = dot::fsils_dot_s(mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) <
        std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) {
      break;
    }
    const double alpha = rho / denom_alpha;

    omp_la::omp_axpby_s(nNo, S, R, -alpha, V);
    apply_schur_operator(S, T);

    const double s_sq = dot::fsils_dot_s(mynNo, lhs.commu, S, S);
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_s(nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    const double t_sq = dot::fsils_dot_s(mynNo, lhs.commu, T, T);
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    const double omega = dot::fsils_dot_s(mynNo, lhs.commu, T, S) / t_sq;

    omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_s(nNo, R, S, -omega, T);

    errO = err;
    err = norm::fsi_ls_norms(mynNo, lhs.commu, R);
    const double rhoO = rho;
    rho = dot::fsils_dot_s(mynNo, lhs.commu, R, Rh);
    const double denom_beta = rhoO * omega;
    if (std::abs(denom_beta) <
        std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) {
      break;
    }
    beta = rho * alpha / denom_beta;

    omp_la::omp_sum_s(nNo, -omega, P, V);
    omp_la::omp_axpby_s(nNo, P, R, beta, P);
    ++i_itr;
  }

  R = X;
  ls.itr = i_itr - 1;
  ls.fNorm = err;
  ls.callD = fsils_cpu_t() - ls.callD;
  ls.dB = (errO < std::numeric_limits<double>::epsilon() || err <= 0.0)
              ? 0.0
              : 10.0 * std::log(err / errO);
  ls.stats.record_call(ls.itr - itr_before,
                       /*restart_cycles=*/1,
                       fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);
  if (err_initial <= ls.absTol) {
    ls.suc = true;
  }
}

void schur_impl(const dsb::MultiConstraintBlockSchurSystem& system,
                fe_fsi_linear_solver::FSILS_subLsType& ls,
                fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
                Array<double>& R)
{
  using namespace fe_fsi_linear_solver;
  auto& lhs = *system.lhs;
  const int mom_ncomp = system.momentum_components;
  const int con_ncomp = system.constraint_components;
  const auto& K = *system.momentum_values;
  const auto& D = *system.D_values;
  const auto& G = *system.G_values;
  const auto& L = *system.L_values;

  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;

  const auto low_rank_profile =
      fe_fsi_linear_solver::distributed_low_rank_correction::inspect(
          lhs, /*mom_start=*/0, mom_ncomp, /*con_start=*/mom_ncomp, con_ncomp);
  const auto strategy =
      fe_fsi_linear_solver::BlockSchurStrategySelector::select(
          lhs, low_rank_profile, con_ncomp);

  if (strategy.use_legacy_scalar_schur()) {
    Vector<double> R_scalar;
    copy_scalar_array_to_vector(R, R_scalar);
    Vector<double> L_scalar(L.ncols());
    for (fsils_int i = 0; i < L.ncols(); ++i) {
      L_scalar(i) = L(0, i);
    }
    schur_face_only_legacy(
        dsb::make_scalar_block_schur_system(lhs, mom_ncomp, K, D, G, L_scalar),
        ls, R_scalar);
    copy_scalar_vector_to_array(R_scalar, R);
    return;
  }

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
  }
  auto& pc = cache.preconditioner;

  auto apply_exact_schur_operator = [&](const Array<double>& in_vec, Array<double>& out_vec) {
    system.GL.apply(
        dso::ghost_synced_input(con_ncomp, in_vec),
        dso::ghost_synced_output(mom_ncomp, GP),
        dso::ghost_synced_output(con_ncomp, SP));
    apply_explicit_constraint_schur_corrections(lhs, pc, in_vec, GP, SP);

    HGP = GP;
    gmres::gmres_v(system.momentum, momentum_ls, HGP);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    if (halo.has_overlap() && !skip_post_solve_overlap_sum()) {
      // The nested momentum solve returns a solution-like overlap vector.
      // Push owner values back to ghosts before reusing it in D*(K^-1*G).
      halo.sync_vector(mom_ncomp, HGP);
    }

    system.D.apply(
        dso::ghost_synced_input(mom_ncomp, HGP),
        dso::ghost_synced_output(con_ncomp, DGP));
    apply_explicit_momentum_schur_corrections(lhs, pc, HGP, DGP);

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

  if (strategy.prefer_schur_gmres) {
    const int restart_dim =
        (low_rank_profile.active_face_corrections > 0)
            ? std::max(1, ls.mItr)
            : std::max(1, std::min((ls.sD > 0 ? ls.sD : ls.mItr), ls.mItr));
    ls.ws.ensure_gmres_v(con_ncomp, nNo, restart_dim);
    auto& h = ls.ws.h;
    auto& u = ls.ws.u3;
    auto& Xg = ls.ws.X2;
    auto& y = ls.ws.y;
    auto& c = ls.ws.c;
    auto& s = ls.ws.s;
    auto& errv = ls.ws.err;

    Array<double> residual(con_ncomp, nNo);
    Array<double> ax(con_ncomp, nNo);

    ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
    ls.suc = false;
    ls.itr = 0;
    Xg = 0.0;

    const double rhs_norm = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, rhs_preconditioned);
    ls.iNorm = rhs_norm;
    const double eps = std::max(ls.absTol, ls.relTol * rhs_norm);
    int restart_cycles = 0;
    int total_itr = 0;

    if (rhs_norm <= ls.absTol) {
      R = Xg;
      ls.fNorm = rhs_norm;
      ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
      ls.dB = 0.0;
      ls.stats.record_call(ls.itr - itr_before,
                           restart_cycles,
                           fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                           pc_setup_time,
                           ls.callD - callD_before);
      return;
    }

    while (total_itr < ls.mItr) {
      ++restart_cycles;

      apply_schur_operator(Xg, ax);
      omp_la::omp_axpby_v(con_ncomp, nNo, residual, rhs_preconditioned, -1.0, ax);
      const double beta = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, residual);
      if (beta < eps) {
        ls.suc = true;
        ls.fNorm = beta;
        break;
      }

      h = 0.0;
      for (int i = 0; i <= restart_dim; ++i) {
        errv(i) = 0.0;
        if (i < restart_dim) {
          c(i) = 0.0;
          s(i) = 0.0;
          y(i) = 0.0;
        }
      }
      errv(0) = beta;

      auto u0 = u.rslice(0);
      u0 = residual;
      omp_la::omp_mul_v(con_ncomp, nNo, 1.0 / beta, u0);

      int last_i = -1;
      bool terminated_inner = false;
      for (int i = 0; i < restart_dim && total_itr < ls.mItr; ++i) {
        last_i = i;
        auto ui = u.rslice(i);
        auto uip1 = u.rslice(i + 1);
        apply_schur_operator(ui, uip1);

        for (int j = 0; j <= i; ++j) {
          h(j, i) = dot::fsils_dot_v(con_ncomp, mynNo, lhs.commu, u.rslice(j), uip1);
          omp_la::omp_sum_v(con_ncomp, nNo, -h(j, i), uip1, u.rslice(j));
        }

        h(i + 1, i) = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, uip1);
        const bool breakdown =
            !(h(i + 1, i) > std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0));
        if (!breakdown) {
          omp_la::omp_mul_v(con_ncomp, nNo, 1.0 / h(i + 1, i), uip1);
        } else {
          h(i + 1, i) = 0.0;
        }

        for (int j = 0; j < i; ++j) {
          const double tmp_h = c(j) * h(j, i) + s(j) * h(j + 1, i);
          h(j + 1, i) = -s(j) * h(j, i) + c(j) * h(j + 1, i);
          h(j, i) = tmp_h;
        }

        const double hypot_h = std::hypot(h(i, i), h(i + 1, i));
        if (hypot_h > std::numeric_limits<double>::epsilon()) {
          c(i) = h(i, i) / hypot_h;
          s(i) = h(i + 1, i) / hypot_h;
        } else {
          c(i) = 1.0;
          s(i) = 0.0;
        }

        h(i, i) = hypot_h;
        h(i + 1, i) = 0.0;
        errv(i + 1) = -s(i) * errv(i);
        errv(i) = c(i) * errv(i);

        ++total_itr;
        ls.itr = total_itr;

        if (std::abs(errv(i + 1)) < eps || breakdown) {
          terminated_inner = true;
          break;
        }
      }

      if (last_i < 0) {
        break;
      }

      for (int i = 0; i <= last_i; ++i) {
        y(i) = errv(i);
      }
      for (int j = last_i; j >= 0; --j) {
        for (int k = j + 1; k <= last_i; ++k) {
          y(j) -= h(j, k) * y(k);
        }
        if (std::abs(h(j, j)) > std::numeric_limits<double>::epsilon()) {
          y(j) /= h(j, j);
        } else {
          y(j) = 0.0;
        }
      }

      for (int j = 0; j <= last_i; ++j) {
        if (std::abs(y(j)) > 1e-30) {
          omp_la::omp_sum_v(con_ncomp, nNo, y(j), Xg, u.rslice(j));
        }
      }

      ls.fNorm = std::abs(errv(last_i + 1));
      if (ls.fNorm < eps) {
        ls.suc = true;
        break;
      }

      if (!terminated_inner) {
        continue;
      }
    }

    if (!ls.suc) {
      apply_schur_operator(Xg, ax);
      omp_la::omp_axpby_v(con_ncomp, nNo, residual, rhs_preconditioned, -1.0, ax);
      ls.fNorm = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, residual);
      ls.suc = (ls.fNorm < eps);
    }

    R = Xg;
    ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
    ls.dB = (rhs_norm < std::numeric_limits<double>::epsilon() || ls.fNorm <= 0.0)
                ? 0.0
                : 10.0 * std::log(ls.fNorm / rhs_norm);
    ls.stats.record_call(ls.itr - itr_before,
                         restart_cycles,
                         fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                         pc_setup_time,
                         ls.callD - callD_before);
    return;
  }

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

void schur_precondition_mc(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                           fe_fsi_linear_solver::FSILS_subLsType& ls,
                           int mom_ncomp, int con_ncomp,
                           const Array<double>& K, const Array<double>& D, const Array<double>& G,
                           const Array<double>& L, Array<double>& R)
{
  schur_precondition_mc(
      dsb::make_multi_constraint_block_schur_system(lhs, mom_ncomp, con_ncomp, K, D, G, L),
      ls, R);
}

void schur_precondition(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                        fe_fsi_linear_solver::FSILS_subLsType& ls,
                        const int nsd,
                        const Array<double>& K, const Array<double>& D, const Array<double>& G,
                        const Vector<double>& L, Vector<double>& R)
{
  Array<double> L_block;
  Array<double> R_block;
  copy_scalar_vector_to_matrix(L, L_block);
  copy_scalar_vector_to_array(R, R_block);
  schur_precondition_mc(lhs, ls, nsd, /*con_ncomp=*/1, K, D, G, L_block, R_block);
  copy_scalar_array_to_vector(R_block, R);
}

void schur_precondition(const fe_fsi_linear_solver::distributed_solver_bundles::ScalarBlockSchurSystem& system,
                        fe_fsi_linear_solver::FSILS_subLsType& ls,
                        Vector<double>& R)
{
  Array<double> L_block;
  Array<double> R_block;
  copy_scalar_vector_to_matrix(*system.L_values, L_block);
  copy_scalar_vector_to_array(R, R_block);
  schur_precondition_mc(
      dsb::make_multi_constraint_block_schur_system(
          *system.lhs, system.momentum_components, /*constraint_components=*/1,
          *system.momentum_values, *system.D_values, *system.G_values, L_block),
      ls, R_block);
  copy_scalar_array_to_vector(R_block, R);
}

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
  schur(dsb::make_scalar_block_schur_system(lhs, nsd, K, D, G, L), ls, momentum_ls, R);
}

void schur(const fe_fsi_linear_solver::distributed_solver_bundles::ScalarBlockSchurSystem& system,
           fe_fsi_linear_solver::FSILS_subLsType& ls,
           fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
           Vector<double>& R)
{
  Array<double> L_block;
  Array<double> R_block;
  copy_scalar_vector_to_matrix(*system.L_values, L_block);
  copy_scalar_vector_to_array(R, R_block);
  schur_impl(
      dsb::make_multi_constraint_block_schur_system(
          *system.lhs, system.momentum_components, /*constraint_components=*/1,
          *system.momentum_values, *system.D_values, *system.G_values, L_block),
      ls, momentum_ls, R_block);
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
  schur_mc(
      dsb::make_multi_constraint_block_schur_system(lhs, mom_ncomp, con_ncomp, K, D, G, L),
      ls, momentum_ls, R);
}

void schur_mc(const fe_fsi_linear_solver::distributed_solver_bundles::MultiConstraintBlockSchurSystem& system,
              fe_fsi_linear_solver::FSILS_subLsType& ls,
              fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
              Array<double>& R)
{
  schur_impl(system, ls, momentum_ls, R);
}

void schur_precondition_mc(const fe_fsi_linear_solver::distributed_solver_bundles::MultiConstraintBlockSchurSystem& system,
                           fe_fsi_linear_solver::FSILS_subLsType& ls,
                           Array<double>& R)
{
  auto& lhs = *system.lhs;
  const fsils_int nNo = lhs.nNo;
  auto& cache = schur_cache_entry(ls);
  if (!cache.valid || cache.mom_ncomp != system.momentum_components ||
      cache.con_ncomp != system.constraint_components || cache.nNo != nNo) {
    cache.preconditioner = build_schur_preconditioner(
        lhs, ls, system.momentum_components, system.constraint_components,
        *system.momentum_values, *system.D_values, *system.G_values, *system.L_values);
    cache.preconditioner.operator_L = &cache.preconditioner.operator_L_storage;
    cache.valid = true;
    cache.mom_ncomp = system.momentum_components;
    cache.con_ncomp = system.constraint_components;
    cache.nNo = nNo;
  }

  apply_schur_preconditioner(
      lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, cache.preconditioner,
      system.constraint_components, nNo, R);
}

} // namespace bicgs
