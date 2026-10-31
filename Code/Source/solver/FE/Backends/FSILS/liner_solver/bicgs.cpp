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
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <sstream>
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
  const HaloExchange halo(lhs);

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
  const CollectiveOps collectives(lhs.commu);
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

    halo.sync_owned_to_ghost_vector(dof, P);
    A.apply(
        dso::ghost_synced_input(dof, P),
        dso::owned_only_output(dof, V));
    double denom_alpha = dot::fsils_dot_v(dof, mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_v(dof, nNo, S, R, -alpha, V);

    halo.sync_owned_to_ghost_vector(dof, S);
    A.apply(
        dso::ghost_synced_input(dof, S),
        dso::owned_only_output(dof, T));
    double schur_locals[3] = {
        norm::fsi_ls_norm_sq_local_v(dof, mynNo, S),
        dot::fsils_nc_dot_v(dof, mynNo, T, T),
        dot::fsils_nc_dot_v(dof, mynNo, T, S),
    };
    double schur_globals[3] = {schur_locals[0], schur_locals[1], schur_locals[2]};
    collectives.allreduce_sum(schur_locals, schur_globals, 3);
    double s_sq = schur_globals[0];
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_v(dof, nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    double t_sq = schur_globals[1];
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_v(dof, nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    double omega = schur_globals[2] / t_sq;

    omp_la::omp_axpbypgz_v(dof, nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_v(dof, nNo, R, S, -omega, T);

    errO = err;
    double update_locals[2] = {
        norm::fsi_ls_norm_sq_local_v(dof, mynNo, R),
        dot::fsils_nc_dot_v(dof, mynNo, R, Rh),
    };
    double update_globals[2] = {update_locals[0], update_locals[1]};
    collectives.allreduce_sum(update_locals, update_globals, 2);
    err = std::sqrt(std::max(0.0, update_globals[0]));
    double rhoO  = rho;
    rho = update_globals[1];
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

  halo.sync_owned_to_ghost_vector(dof, X);
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
  const HaloExchange halo(lhs);

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
  const CollectiveOps collectives(lhs.commu);
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

    halo.sync_owned_to_ghost_scalar(P);
    A.apply(
        dso::ghost_synced_input(P),
        dso::owned_only_output(V));
    double denom_alpha = dot::fsils_dot_s(mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_s(nNo, S, R, -alpha, V);

    halo.sync_owned_to_ghost_scalar(S);
    A.apply(
        dso::ghost_synced_input(S),
        dso::owned_only_output(T));
    double schur_locals[3] = {
        norm::fsi_ls_norm_sq_local_s(mynNo, S),
        dot::fsils_nc_dot_s(mynNo, T, T),
        dot::fsils_nc_dot_s(mynNo, T, S),
    };
    double schur_globals[3] = {schur_locals[0], schur_locals[1], schur_locals[2]};
    collectives.allreduce_sum(schur_locals, schur_globals, 3);
    double s_sq = schur_globals[0];
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_s(nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    double t_sq = schur_globals[1];
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    double omega = schur_globals[2] / t_sq;

    omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_s(nNo, R, S, -omega, T);

    errO = err;
    double update_locals[2] = {
        norm::fsi_ls_norm_sq_local_s(mynNo, R),
        dot::fsils_nc_dot_s(mynNo, R, Rh),
    };
    double update_globals[2] = {update_locals[0], update_locals[1]};
    collectives.allreduce_sum(update_locals, update_globals, 2);
    err = std::sqrt(std::max(0.0, update_globals[0]));
    double rhoO  = rho;
    rho = update_globals[1];
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

  halo.sync_owned_to_ghost_scalar(X);
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

[[nodiscard]] bool env_enabled(const char* name) noexcept
{
  const char* env = std::getenv(name);
  if (env == nullptr) {
    return false;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  return *env != '\0' && *env != '0';
}

[[nodiscard]] int forced_schur_bicgstab_iterations(
    const fe_fsi_linear_solver::FSILS_subLsType& ls) noexcept
{
  const int requested =
      parse_int_env("SVMP_FSILS_BLOCKSCHUR_SCHUR_FIXED_ITERS", 0);
  if (requested <= 0) {
    return 0;
  }
  return std::min(requested, std::max(0, ls.mItr));
}

[[nodiscard]] int schur_preconditioner_reuse_solve_limit() noexcept
{
  static const int limit = [] {
    const int requested = parse_int_env("SVMP_FSILS_BLOCKSCHUR_SCHUR_PC_REUSE_SOLVES", 1);
    return std::max(-1, requested);
  }();
  return limit;
}

[[nodiscard]] int forced_schur_preconditioner_code() noexcept
{
  static const int forced = []() noexcept {
    const char* env = std::getenv("SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_PC");
    if (env == nullptr) {
      return -1;
    }

    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
      return std::isspace(c) != 0;
    }), value.end());

    using fe_fsi_linear_solver::SchurPreconditionerType;
    if (value == "diag-l") {
      return static_cast<int>(SchurPreconditionerType::DIAG_L);
    }
    if (value == "blockdiag-l") {
      return static_cast<int>(SchurPreconditionerType::BLOCKDIAG_L);
    }
    if (value == "ilu-l") {
      return static_cast<int>(SchurPreconditionerType::ILU_L);
    }
    if (value == "algebraic-shat") {
      return static_cast<int>(SchurPreconditionerType::ALGEBRAIC_SHAT);
    }
    return -1;
  }();
  return forced;
}

[[nodiscard]] bool schur_partition_coarse_modes_enabled() noexcept
{
  return env_enabled("SVMP_FSILS_SCHUR_PARTITION_COARSE") &&
         !env_enabled("SVMP_FSILS_DISABLE_SCHUR_PARTITION_COARSE");
}

[[nodiscard]] bool schur_global_mean_coarse_mode_enabled() noexcept
{
  return env_enabled("SVMP_FSILS_SCHUR_GLOBAL_MEAN_COARSE") &&
         !env_enabled("SVMP_FSILS_DISABLE_SCHUR_GLOBAL_MEAN_COARSE");
}

[[nodiscard]] const char* schur_dump_prefix() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_SCHUR_DUMP_PREFIX");
  if (env == nullptr) {
    return nullptr;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  return (*env == '\0') ? nullptr : env;
}

[[nodiscard]] int schur_dump_max_solves() noexcept
{
  return parse_int_env("SVMP_FSILS_SCHUR_DUMP_MAX_SOLVES", 1);
}

[[nodiscard]] int schur_dump_max_iters() noexcept
{
  return std::max(0, parse_int_env("SVMP_FSILS_SCHUR_DUMP_MAX_ITERS", 1));
}

void dump_schur_array_owned(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                            const char* prefix,
                            std::uint64_t solve_id,
                            const char* label,
                            const Array<double>& values)
{
  if (prefix == nullptr || *prefix == '\0') {
    return;
  }

  std::ostringstream path;
  path << prefix
       << ".solve" << solve_id
       << "." << label
       << ".rank" << lhs.commu.task
       << ".txt";
  std::ofstream out(path.str());
  if (!out) {
    return;
  }

  const int con_ncomp = values.nrows();
  const bool has_debug_global_nodes = lhs.debug_global_nodes.size() == lhs.nNo;

  out << std::setprecision(17) << std::scientific;
  out << "# task " << lhs.commu.task
      << " nTasks " << lhs.commu.nTasks
      << " solve " << solve_id
      << " label " << label
      << " con_ncomp " << con_ncomp
      << " mynNo " << lhs.mynNo
      << " nNo " << lhs.nNo << '\n';
  out << "# global_node component value old_node internal_node";
  if (has_debug_global_nodes) {
    out << " backend_node";
  }
  out << '\n';

  for (fsils_int old_node = 0; old_node < lhs.nNo; ++old_node) {
    if (old_node < 0 || old_node >= lhs.gNodes.size()) {
      continue;
    }
    const int internal_node = lhs.map(old_node);
    if (internal_node < 0 || internal_node >= lhs.mynNo) {
      continue;
    }
    const int global_node =
        has_debug_global_nodes ? lhs.debug_global_nodes(old_node) : lhs.gNodes(old_node);
    if (global_node < 0) {
      continue;
    }
    for (int comp = 0; comp < con_ncomp; ++comp) {
      out << global_node << ' '
          << comp << ' '
          << values(comp, internal_node) << ' '
          << old_node << ' '
          << internal_node;
      if (has_debug_global_nodes) {
        out << ' ' << lhs.gNodes(old_node);
      }
      out << '\n';
    }
  }
}

void dump_schur_vector_owned(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                             const char* prefix,
                             std::uint64_t solve_id,
                             const char* label,
                             const Vector<double>& values)
{
  if (prefix == nullptr || *prefix == '\0') {
    return;
  }

  std::ostringstream path;
  path << prefix
       << ".legacy.solve" << solve_id
       << "." << label
       << ".rank" << lhs.commu.task
       << ".txt";
  std::ofstream out(path.str());
  if (!out) {
    return;
  }

  const bool has_debug_global_nodes = lhs.debug_global_nodes.size() == lhs.nNo;

  out << std::setprecision(17) << std::scientific;
  out << "# task " << lhs.commu.task
      << " nTasks " << lhs.commu.nTasks
      << " solve " << solve_id
      << " label " << label
      << " mynNo " << lhs.mynNo
      << " nNo " << lhs.nNo << '\n';
  out << "# global_node value old_node internal_node";
  if (has_debug_global_nodes) {
    out << " backend_node";
  }
  out << '\n';

  for (fsils_int old_node = 0; old_node < lhs.nNo; ++old_node) {
    if (old_node < 0 || old_node >= lhs.gNodes.size()) {
      continue;
    }
    const int internal_node = lhs.map(old_node);
    if (internal_node < 0 || internal_node >= lhs.mynNo || internal_node >= values.size()) {
      continue;
    }
    const int global_node =
        has_debug_global_nodes ? lhs.debug_global_nodes(old_node) : lhs.gNodes(old_node);
    if (global_node < 0) {
      continue;
    }
    out << global_node << ' '
        << values(internal_node) << ' '
        << old_node << ' '
        << internal_node;
    if (has_debug_global_nodes) {
      out << ' ' << lhs.gNodes(old_node);
    }
    out << '\n';
  }
}

[[nodiscard]] std::uint64_t hash_mix(std::uint64_t seed, std::uint64_t value) noexcept
{
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
  return seed;
}

[[nodiscard]] std::uint64_t hash_double_bits(std::uint64_t seed, double value) noexcept
{
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double and uint64_t sizes must match");
  std::memcpy(&bits, &value, sizeof(bits));
  return hash_mix(seed, bits);
}

[[nodiscard]] std::uint64_t hash_array_values(std::uint64_t seed, const Array<double>& values) noexcept
{
  seed = hash_mix(seed, static_cast<std::uint64_t>(values.nrows()));
  seed = hash_mix(seed, static_cast<std::uint64_t>(values.ncols()));
  seed = hash_mix(seed, static_cast<std::uint64_t>(values.size()));
  const double* data = values.data();
  if (data == nullptr) {
    return seed;
  }
  for (int i = 0; i < values.size(); ++i) {
    seed = hash_double_bits(seed, data[i]);
  }
  return seed;
}

[[nodiscard]] std::uint64_t hash_vector_values(std::uint64_t seed,
                                               const std::vector<double>& values) noexcept
{
  seed = hash_mix(seed, static_cast<std::uint64_t>(values.size()));
  for (double value : values) {
    seed = hash_double_bits(seed, value);
  }
  return seed;
}

[[nodiscard]] std::uint64_t schur_cache_topology_signature(const fe_fsi_linear_solver::FSILS_lhsType& lhs) noexcept
{
  std::uint64_t signature = 0xcbf29ce484222325ULL;
  signature = hash_mix(signature, static_cast<std::uint64_t>(lhs.nFaces));
  signature = hash_mix(signature, static_cast<std::uint64_t>(lhs.reduced_updates.size()));
  signature = hash_mix(signature, static_cast<std::uint64_t>(lhs.grouped_bordered_field_couplings.size()));
  signature = hash_mix(signature, static_cast<std::uint64_t>(lhs.native_face_rank_one_count));

  for (const auto& face : lhs.face) {
    if (!face.coupledFlag || face.nNo <= 0) {
      continue;
    }
    signature = hash_mix(signature, static_cast<std::uint64_t>(face.dof));
    signature = hash_mix(signature, static_cast<std::uint64_t>(face.nNo));
    signature = hash_mix(signature, static_cast<std::uint64_t>(face.sharedFlag ? 1 : 0));
    for (int i = 0; i < face.nNo; ++i) {
      signature = hash_mix(signature, static_cast<std::uint64_t>(face.glob(i) + 1));
    }
  }

  for (const auto& update : lhs.reduced_updates) {
    signature = hash_mix(signature, static_cast<std::uint64_t>(update.grouped_coupling_id + 7));
    signature = hash_mix(signature, static_cast<std::uint64_t>(update.active_components.size()));
    for (const int comp : update.active_components) {
      signature = hash_mix(signature, static_cast<std::uint64_t>(comp + 11));
    }
    signature = hash_mix(signature, static_cast<std::uint64_t>(update.left.size()));
    for (const auto& entry : update.left) {
      signature = hash_mix(signature, static_cast<std::uint64_t>(entry.node + 13));
      signature = hash_mix(signature, static_cast<std::uint64_t>(entry.full_component + 17));
    }
    signature = hash_mix(signature, static_cast<std::uint64_t>(update.right.size()));
    for (const auto& entry : update.right) {
      signature = hash_mix(signature, static_cast<std::uint64_t>(entry.node + 19));
      signature = hash_mix(signature, static_cast<std::uint64_t>(entry.full_component + 23));
    }
  }

  for (const auto& group : lhs.grouped_bordered_field_couplings) {
    signature = hash_mix(signature, static_cast<std::uint64_t>(group.grouped_coupling_id + 29));
    signature = hash_mix(signature, static_cast<std::uint64_t>(group.modes.size()));
    for (const auto& mode : group.modes) {
      signature = hash_mix(signature, static_cast<std::uint64_t>(mode.active_components.size()));
      for (const int comp : mode.active_components) {
        signature = hash_mix(signature, static_cast<std::uint64_t>(comp + 31));
      }
      signature = hash_mix(signature, static_cast<std::uint64_t>(mode.left.size()));
      for (const auto& entry : mode.left) {
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.node + 37));
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.full_component + 41));
      }
      signature = hash_mix(signature, static_cast<std::uint64_t>(mode.right.size()));
      for (const auto& entry : mode.right) {
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.node + 43));
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.full_component + 47));
      }
    }
  }

  return signature;
}

[[nodiscard]] std::uint64_t schur_cache_operator_signature(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                                           const Array<double>& K,
                                                           const Array<double>& D,
                                                           const Array<double>& G,
                                                           const Array<double>& L) noexcept
{
  std::uint64_t signature = 0x6eed0e9da4d94a4fULL;
  signature = hash_array_values(signature, K);
  signature = hash_array_values(signature, D);
  signature = hash_array_values(signature, G);
  signature = hash_array_values(signature, L);
  signature = hash_vector_values(signature, lhs.native_face_pc_dense_coeff);
  signature = hash_vector_values(signature, lhs.reduced_update_pc_inner_inv);
  for (const auto& group : lhs.grouped_bordered_field_couplings) {
    signature = hash_vector_values(signature, group.aux_matrix);
    for (const auto& mode : group.modes) {
      for (const auto& entry : mode.left) {
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.node + 101));
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.full_component + 107));
        signature = hash_double_bits(signature, entry.value);
      }
      for (const auto& entry : mode.right) {
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.node + 103));
        signature = hash_mix(signature, static_cast<std::uint64_t>(entry.full_component + 109));
        signature = hash_double_bits(signature, entry.value);
      }
    }
  }
  return signature;
}

[[nodiscard]] bool skip_post_solve_halo_sync() noexcept
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

void refresh_scalar_ghosts(const fe_fsi_linear_solver::HaloExchange& halo,
                           Vector<double>& values,
                           bool skip_refresh = false)
{
  if (skip_refresh || !halo.has_owned_halo()) {
    return;
  }
  halo.sync_owned_to_ghost_scalar(values);
}

void refresh_vector_ghosts(const fe_fsi_linear_solver::HaloExchange& halo,
                           int dof,
                           Array<double>& values,
                           bool skip_refresh = false)
{
  if (skip_refresh || !halo.has_owned_halo()) {
    return;
  }
  halo.sync_owned_to_ghost_vector(dof, values);
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
  std::vector<Array<double>> coarse_right;
  std::vector<Array<double>> coarse_preconditioned_left;
  std::vector<double> coarse_inner_inv;
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
  fsils_int nnz{0};
  std::uint64_t topology_signature{0};
  std::uint64_t operator_signature{0};
  fe_fsi_linear_solver::SchurPreconditionerType schur_preconditioner{
      fe_fsi_linear_solver::SchurPreconditionerType::ALGEBRAIC_SHAT};
  fe_fsi_linear_solver::SchurMomentumApproximationType momentum_approximation{
      fe_fsi_linear_solver::SchurMomentumApproximationType::ILU_K};
  int solves_since_build{0};
  SchurPreconditionerData preconditioner;
};

[[nodiscard]] bool should_rebuild_schur_cache(const SchurSolveCacheEntry& cache,
                                              const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                              const fe_fsi_linear_solver::FSILS_subLsType& ls,
                                              int mom_ncomp,
                                              int con_ncomp,
                                              std::uint64_t topology_signature,
                                              std::uint64_t operator_signature) noexcept
{
  if (!cache.valid || cache.mom_ncomp != mom_ncomp || cache.con_ncomp != con_ncomp ||
      cache.nNo != lhs.nNo || cache.nnz != lhs.nnz) {
    return true;
  }
  if (cache.topology_signature != topology_signature ||
      cache.operator_signature != operator_signature ||
      cache.schur_preconditioner != ls.schur_preconditioner ||
      cache.momentum_approximation != ls.schur_momentum_approximation) {
    return true;
  }
  const int reuse_limit = schur_preconditioner_reuse_solve_limit();
  return reuse_limit >= 0 && cache.solves_since_build > reuse_limit;
}

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

void apply_hat_schur_operator(const Array<fsils_int>& rowPtr,
                              const Vector<fsils_int>& colPtr,
                              const Vector<fsils_int>& diagPtr,
                              fe_fsi_linear_solver::FSILS_lhsType& lhs,
                              SchurPreconditionerData& pc,
                              const Array<double>& in_vec,
                              Array<double>& out_vec);

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
                                const fe_fsi_linear_solver::FSILS_lhsType* lhs,
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
  Array<double> diag_blocks(block_entries, nNo);
  for (fsils_int i = 0; i < nNo; ++i) {
    const fsils_int diag_nz = diagPtr(i);
    for (int entry = 0; entry < block_entries; ++entry) {
      diag_blocks(entry, i) = values(entry, diag_nz);
    }
  }
  if (lhs != nullptr) {
    const fe_fsi_linear_solver::HaloExchange halo(*lhs);
    halo.sync_owned_to_ghost_vector(block_entries, diag_blocks);
  }
  for (fsils_int i = 0; i < nNo; ++i) {
    double* out = block_ptr(inv_blocks, block_entries, i);
    if (diagonal_only) {
      set_zero(out, block_entries);
      for (int d = 0; d < block_size; ++d) {
        out[d * block_size + d] = safe_inverse(
            diag_blocks(d * block_size + d, i));
      }
      continue;
    }

    for (int entry = 0; entry < block_entries; ++entry) {
      diag_block[static_cast<std::size_t>(entry)] = diag_blocks(entry, i);
    }
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

void sync_schur_halo(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                        int con_ncomp,
                        Array<double>& x)
{
  const fe_fsi_linear_solver::HaloExchange halo(lhs);
  halo.sync_owned_to_ghost_vector(con_ncomp, x, skip_post_solve_halo_sync());
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

const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& projected_left_entries(
    const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update)
{
  // Match the exact reduced operator: owned-only on the contracting right
  // side, full local support on the scattered left side.
  return !update.left_scaled.empty()     ? update.left_scaled
      : !update.left.empty()             ? update.left
      : !update.left_scaled_owned.empty() ? update.left_scaled_owned
                                         : update.left_owned;
}

const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& projected_right_entries(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update)
{
  if (!update.right_scaled_owned.empty()) {
    return update.right_scaled_owned;
  }
  if (!update.right_owned.empty()) {
    return update.right_owned;
  }
  if (lhs.commu.nTasks > 1) {
    static const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry> empty_entries;
    return empty_entries;
  }
  return !update.right_scaled.empty() ? update.right_scaled : update.right;
}

const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& projected_right_entries_for_transpose(
    const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update)
{
  // The exact reduced Schur G^T apply needs the full local support of the
  // right factor so each rank can contribute to the constraint rows it owns.
  // Using the owned-only right entries here drops valid ghost-supported rows
  // in distributed runs and can zero the resulting schur_right column.
  return !update.right_scaled.empty()     ? update.right_scaled
      : !update.right.empty()             ? update.right
      : !update.right_scaled_owned.empty() ? update.right_scaled_owned
                                          : update.right_owned;
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

void trace_dense_matrix(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                        const char* label,
                        const std::vector<double>& values,
                        int rank)
{
  if (!env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING")) {
    return;
  }
  if (!lhs.commu.masF) {
    return;
  }
  std::fprintf(stderr,
               "[BICGS_SCHUR_SETUP] %s rank=%d",
               label,
               rank);
  for (int i = 0; i < rank; ++i) {
    for (int j = 0; j < rank; ++j) {
      std::fprintf(stderr,
                   " m(%d,%d)=%.17e",
                   i,
                   j,
                   values[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                          static_cast<std::size_t>(j)]);
    }
  }
  std::fprintf(stderr, "\n");
}

double dense_dense_owned_norm_local(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                    int dof,
                                    const Array<double>& vec)
{
  const double local = dense_dense_owned_dot_local(lhs, dof, vec, vec);
  return (local > 0.0) ? std::sqrt(local) : 0.0;
}

void trace_dense_vector_owned_entries(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                      const char* label,
                                      int dof,
                                      const Array<double>& vec)
{
  if (!env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS")) {
    return;
  }
  static int dump_count = 0;
  if (dump_count >= 16) {
    return;
  }
  ++dump_count;
  std::fprintf(stderr, "[BICGS_SCHUR_SETUP] %s rank=%d", label, lhs.commu.task);
  for (fsils_int old = 0; old < lhs.nNo; ++old) {
    if (old < 0 || old >= lhs.gNodes.size()) {
      continue;
    }
    const int internal = lhs.map(old);
    if (internal < 0 || internal >= lhs.mynNo) {
      continue;
    }
    for (int comp = 0; comp < dof; ++comp) {
      const double value = vec(comp, internal);
      if (std::abs(value) <= 1e-30) {
        continue;
      }
      std::fprintf(stderr,
                   " entry(old=%lld,global=%d,internal=%d,comp=%d,val=%.17e)",
                   static_cast<long long>(old),
                   lhs.gNodes(old),
                   internal,
                   comp,
                   value);
    }
  }
  std::fprintf(stderr, "\n");
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
  const bool trace_setup = env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING");
  col_values.resize(in_dof, nNo);
  col_values = 0.0;

  std::vector<double> contrib(static_cast<size_t>(in_dof), 0.0);
  bool reported_oob = false;
  for (fsils_int row = 0; row < nNo; ++row) {
    const double* row_vec = row_values.data() + static_cast<size_t>(row) * static_cast<size_t>(out_dof);
    for (fsils_int p = rowPtr(0, row); p <= rowPtr(1, row); ++p) {
      const fsils_int col = colPtr(p);
      if (col < 0 || col >= nNo) {
        if (trace_setup && !reported_oob) {
          std::fprintf(stderr,
                       "[BICGS_SCHUR_SETUP] multiply_rect_transpose_local out_of_range "
                       "row=%lld p=%lld col=%lld nNo=%lld\n",
                       static_cast<long long>(row),
                       static_cast<long long>(p),
                       static_cast<long long>(col),
                       static_cast<long long>(nNo));
          reported_oob = true;
        }
        continue;
      }
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
  const bool trace_setup =
      env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING") && lhs.commu.masF;
  const double total_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  hat.low_rank_right.clear();
  hat.low_rank_preconditioned_left.clear();
  hat.low_rank_left.clear();
  hat.low_rank_preconditioned_right_t.clear();
  hat.low_rank_inner_inv.clear();
  hat.low_rank_inner_inv_t.clear();

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
        expanded[static_cast<std::size_t>(old_rank + i) *
                     static_cast<std::size_t>(old_rank + block_rank) +
                 static_cast<std::size_t>(old_rank + j)] =
            block[static_cast<std::size_t>(i) * static_cast<std::size_t>(block_rank) +
                  static_cast<std::size_t>(j)];
      }
    }
    dense_seed.swap(expanded);
  };

  std::vector<double> dense_seed;
  dense_seed.reserve(lhs.reduced_updates.size());

  auto append_reduced_mode = [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update)
      -> bool {
    Array<double> left_mode;
    Array<double> right_mode;
    const auto& left_entries = projected_left_entries(update);
    const auto& right_entries = projected_right_entries(lhs, update);
    const bool have_left_local =
        fill_projected_reduced_vector(lhs, update, left_entries, mom_ncomp, left_mode);
    const bool have_right_local =
        fill_projected_reduced_vector(lhs, update, right_entries, mom_ncomp, right_mode);
    int have_left = have_left_local ? 1 : 0;
    int have_right = have_right_local ? 1 : 0;
    if (lhs.commu.nTasks > 1) {
      const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
      collectives.allreduce_sum(have_left, have_left);
      collectives.allreduce_sum(have_right, have_right);
    }
    if (have_left == 0 || have_right == 0) {
      return false;
    }

    hat.low_rank_left.push_back(std::move(left_mode));
    hat.low_rank_right.push_back(std::move(right_mode));
    return true;
  };

  auto append_face_mode = [&](const fe_fsi_linear_solver::FSILS_faceType& face) -> bool {
    Array<double> mode;
    const bool have_mode_local =
        fill_projected_face_vector(lhs, face, mom_ncomp, mode);
    int have_mode = have_mode_local ? 1 : 0;
    if (lhs.commu.nTasks > 1) {
      const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
      collectives.allreduce_sum(have_mode, have_mode);
    }
    if (have_mode == 0) {
      return false;
    }

    hat.low_rank_left.push_back(mode);
    hat.low_rank_right.push_back(std::move(mode));
    return true;
  };

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
    const int old_rank = static_cast<int>(hat.low_rank_left.size());
    if (!append_reduced_mode(update)) {
      continue;
    }
    append_dense_seed_block(dense_seed,
                            old_rank,
                            std::vector<double>{safe_inverse(update.sigma)},
                            /*block_rank=*/1);
  }

  for (const auto& face : lhs.face) {
    if (!face.coupledFlag || std::abs(face.res) <= 1e-30 || face.nNo <= 0) {
      continue;
    }
    const int old_rank = static_cast<int>(hat.low_rank_left.size());
    if (!append_face_mode(face)) {
      continue;
    }
    append_dense_seed_block(dense_seed,
                            old_rank,
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

    const int old_rank = static_cast<int>(hat.low_rank_left.size());
    int appended = 0;
    for (const auto& mode : group.modes) {
      if (append_reduced_mode(mode)) {
        appended += 1;
      }
    }
    if (appended != block_rank) {
      hat.low_rank_left.resize(static_cast<std::size_t>(old_rank));
      hat.low_rank_right.resize(static_cast<std::size_t>(old_rank));
      continue;
    }

    std::vector<double> seed_block(group.aux_matrix.size(), 0.0);
    for (std::size_t idx = 0; idx < group.aux_matrix.size(); ++idx) {
      seed_block[idx] = -group.aux_matrix[idx];
    }
    append_dense_seed_block(dense_seed, old_rank, seed_block, block_rank);
  }

  const int total_rank = static_cast<int>(hat.low_rank_left.size());
  if (total_rank <= 0 ||
      dense_seed.size() != static_cast<std::size_t>(total_rank * total_rank)) {
    hat.low_rank_right.clear();
    hat.low_rank_left.clear();
    if (trace_setup) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] grouped_momentum_hat rank=0 early_exit=1 total=%e\n",
                   fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
    }
    return;
  }

  const double precondition_left_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  hat.low_rank_preconditioned_left = hat.low_rank_left;
  hat.low_rank_preconditioned_right_t = hat.low_rank_right;
  for (auto& vec : hat.low_rank_preconditioned_left) {
    apply_momentum_hat(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat, vec);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    halo.sync_owned_to_ghost_vector(mom_ncomp, vec, skip_post_solve_halo_sync());
  }
  for (auto& vec : hat.low_rank_preconditioned_right_t) {
    apply_momentum_hat_transpose(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, hat, vec);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    halo.sync_owned_to_ghost_vector(mom_ncomp, vec, skip_post_solve_halo_sync());
  }
  const double precondition_left_t =
      trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - precondition_left_t0 : 0.0;

  const double dense_dot_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  std::vector<double> dense_m(static_cast<std::size_t>(total_rank) *
                                  static_cast<std::size_t>(total_rank),
                              0.0);
  std::vector<double> dense_mt(static_cast<std::size_t>(total_rank) *
                                   static_cast<std::size_t>(total_rank),
                               0.0);
  for (int i = 0; i < total_rank; ++i) {
    for (int j = 0; j < total_rank; ++j) {
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
              static_cast<std::size_t>(j)] =
          dense_dense_owned_dot_local(
              lhs,
              mom_ncomp,
              hat.low_rank_right[static_cast<std::size_t>(i)],
              hat.low_rank_preconditioned_left[static_cast<std::size_t>(j)]);
      dense_mt[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
               static_cast<std::size_t>(j)] =
          dense_dense_owned_dot_local(
              lhs,
              mom_ncomp,
              hat.low_rank_left[static_cast<std::size_t>(i)],
              hat.low_rank_preconditioned_right_t[static_cast<std::size_t>(j)]);
    }
  }
  allreduce_sum_in_place(const_cast<fe_fsi_linear_solver::FSILS_lhsType&>(lhs), dense_m);
  allreduce_sum_in_place(const_cast<fe_fsi_linear_solver::FSILS_lhsType&>(lhs), dense_mt);
  trace_dense_matrix(lhs, "grouped_momentum_hat_dense_m_preseed", dense_m, total_rank);
  trace_dense_matrix(lhs, "grouped_momentum_hat_dense_mt_preseed", dense_mt, total_rank);
  for (int i = 0; i < total_rank; ++i) {
    for (int j = 0; j < total_rank; ++j) {
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
              static_cast<std::size_t>(j)] +=
          dense_seed[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
                     static_cast<std::size_t>(j)];
      dense_mt[static_cast<std::size_t>(i) * static_cast<std::size_t>(total_rank) +
               static_cast<std::size_t>(j)] +=
          dense_seed[static_cast<std::size_t>(j) * static_cast<std::size_t>(total_rank) +
                     static_cast<std::size_t>(i)];
    }
  }
  trace_dense_matrix(lhs, "grouped_momentum_hat_dense_m_total", dense_m, total_rank);
  trace_dense_matrix(lhs, "grouped_momentum_hat_dense_mt_total", dense_mt, total_rank);
  const double dense_dot_t =
      trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - dense_dot_t0 : 0.0;

  const double invert_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
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
  if (trace_setup) {
    const double invert_t = fe_fsi_linear_solver::fsils_cpu_t() - invert_t0;
    int active_groups = 0;
    int active_faces = 0;
    for (const auto& group : lhs.grouped_bordered_field_couplings) {
      if (group.active && !group.modes.empty()) {
        active_groups += 1;
      }
    }
    for (const auto& face : lhs.face) {
      if (face.coupledFlag && std::abs(face.res) > 1e-30 && face.nNo > 0) {
        active_faces += 1;
      }
    }
    std::fprintf(stderr,
                 "[BICGS_SCHUR_SETUP] grouped_momentum_hat rank=%d groups=%d faces=%d reduced=%zu "
                 "precondition_vectors=%e dense_dots=%e invert=%e total=%e\n",
                 total_rank,
                 active_groups,
                 active_faces,
                 lhs.reduced_updates.size(),
                 precondition_left_t,
                 dense_dot_t,
                 invert_t,
                 fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
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

  const bool trace_setup =
      env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING") && lhs.commu.masF;
  const double total_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  MomentumHatData hat{};
  Array<double> K_eff = K;
  const auto momentum_operator =
      dso::SparseOperatorBundle(lhs, rowPtr, colPtr).vector(mom_ncomp, K_eff);
  const bool has_grouped_bordered = !lhs.grouped_bordered_field_couplings.empty();
  const bool has_active_grouped_bordered = std::any_of(
      lhs.grouped_bordered_field_couplings.begin(),
      lhs.grouped_bordered_field_couplings.end(),
      [](const fe_fsi_linear_solver::FSILS_groupedBorderedFieldCouplingType& group) {
        return group.active && !group.modes.empty();
      });
  const bool has_active_face_corrections = std::any_of(
      lhs.face.begin(),
      lhs.face.end(),
      [](const fe_fsi_linear_solver::FSILS_faceType& face) {
        return face.coupledFlag && std::abs(face.res) > 1e-30 && face.nNo > 0;
      });
  const bool has_active_standalone_reduced_updates = std::any_of(
      lhs.reduced_updates.begin(),
      lhs.reduced_updates.end(),
      [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update) {
        if (!update.active || std::abs(update.sigma) <= 1e-30 ||
            update.grouped_coupling_id == kNativeFaceDuplicateCouplingId) {
          return false;
        }
        return !(has_grouped_bordered && update.grouped_coupling_id >= 0);
      });
  const bool has_exact_momentum_modes =
      has_active_grouped_bordered ||
      has_active_face_corrections ||
      has_active_standalone_reduced_updates;

  auto build_candidate =
      [&](SchurMomentumApproximationType candidate_approx) -> MomentumHatData {
    MomentumHatData candidate{};
    candidate.mynNo = lhs.mynNo;
    candidate.commu = &lhs.commu;
    switch (candidate_approx) {
      case SchurMomentumApproximationType::DIAG_K:
        build_point_inverse_blocks(diagPtr, &lhs, mom_ncomp, nNo, K_eff, /*diagonal_only=*/true, candidate.point_inv);
        build_sparse_block_diagonal_inverse(diagPtr, mom_ncomp, nNo, K_eff.ncols(), candidate.point_inv,
                                            candidate.sparse_inverse);
        build_grouped_momentum_hat_low_rank_correction(rowPtr, colPtr, diagPtr, lhs,
                                                       mom_ncomp, nNo, candidate);
        return candidate;
      case SchurMomentumApproximationType::BLOCKDIAG_K:
        build_point_inverse_blocks(diagPtr, &lhs, mom_ncomp, nNo, K_eff, /*diagonal_only=*/false, candidate.point_inv);
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
        build_point_inverse_blocks(diagPtr, &lhs, mom_ncomp, nNo, K_eff, /*diagonal_only=*/false, candidate.point_inv);
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
    const fe_fsi_linear_solver::HaloExchange halo(lhs);

    auto accumulate_mode_score = [&](const Array<double>& left_mode,
                                     const Array<double>& right_mode) {
      Array<double> approx_apply = left_mode;
      Array<double> residual(mom_ncomp, nNo);
      apply_momentum_hat(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, candidate, approx_apply);
      halo.sync_owned_to_ghost_vector(mom_ncomp, approx_apply, skip_post_solve_halo_sync());
      momentum_operator.apply(
          dso::ghost_synced_input(mom_ncomp, approx_apply),
          dso::owned_only_output(mom_ncomp, residual));
      omp_la::omp_axpby_v(mom_ncomp, nNo, residual, residual, -1.0, left_mode);
      const double den = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                          const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                          left_mode, left_mode);
      const double num = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                          const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                          residual, residual);
      denominator += static_cast<long double>(den);
      numerator += static_cast<long double>(num);

      Array<double> approx_apply_t = right_mode;
      Array<double> residual_t(mom_ncomp, nNo);
      apply_momentum_hat_transpose(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, candidate, approx_apply_t);
      halo.sync_owned_to_ghost_vector(mom_ncomp, approx_apply_t, skip_post_solve_halo_sync());
      multiply_rect_transpose_local(rowPtr, colPtr, nNo,
                                    mom_ncomp, mom_ncomp, K_eff,
                                    approx_apply_t, residual_t);
      halo.sync_owned_to_ghost_vector(mom_ncomp, residual_t, skip_post_solve_halo_sync());
      omp_la::omp_axpby_v(mom_ncomp, nNo, residual_t, residual_t, -1.0, right_mode);
      const double den_t = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                            const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                            right_mode, right_mode);
      const double num_t = dot::fsils_dot_v(mom_ncomp, lhs.mynNo,
                                            const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu),
                                            residual_t, residual_t);
      denominator += static_cast<long double>(den_t);
      numerator += static_cast<long double>(num_t);
    };

    auto score_reduced_mode =
        [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& mode) {
      Array<double> left_mode;
      Array<double> right_mode;
      const bool have_left_local =
          fill_projected_reduced_vector(lhs, mode, projected_left_entries(mode), mom_ncomp, left_mode);
      const bool have_right_local =
          fill_projected_reduced_vector(lhs, mode, projected_right_entries(lhs, mode), mom_ncomp, right_mode);
      int have_left = have_left_local ? 1 : 0;
      int have_right = have_right_local ? 1 : 0;
      if (lhs.commu.nTasks > 1) {
        const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
        collectives.allreduce_sum(have_left, have_left);
        collectives.allreduce_sum(have_right, have_right);
      }
      if (have_left == 0 || have_right == 0) {
        return;
      }
      accumulate_mode_score(left_mode, right_mode);
    };

    for (const auto& update : lhs.reduced_updates) {
      if (!update.active || std::abs(update.sigma) <= 1e-30 ||
          update.grouped_coupling_id == kNativeFaceDuplicateCouplingId) {
        continue;
      }
      if (has_grouped_bordered && update.grouped_coupling_id >= 0) {
        continue;
      }
      score_reduced_mode(update);
    }

    for (const auto& face : lhs.face) {
      if (!face.coupledFlag || std::abs(face.res) <= 1e-30 || face.nNo <= 0) {
        continue;
      }
      Array<double> mode;
      const bool have_mode_local =
          fill_projected_face_vector(lhs, face, mom_ncomp, mode);
      int have_mode = have_mode_local ? 1 : 0;
      if (lhs.commu.nTasks > 1) {
        const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
        collectives.allreduce_sum(have_mode, have_mode);
      }
      if (have_mode == 0) {
        continue;
      }
      accumulate_mode_score(mode, mode);
    }

    for (const auto& group : lhs.grouped_bordered_field_couplings) {
      if (!group.active) {
        continue;
      }
      for (const auto& mode : group.modes) {
        score_reduced_mode(mode);
      }
    }

    if (!(denominator > 1e-30L)) {
      return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(numerator / denominator);
  };

  if (has_exact_momentum_modes && approx == SchurMomentumApproximationType::ILU_K) {
    const double ilu_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    MomentumHatData ilu_hat = build_candidate(SchurMomentumApproximationType::ILU_K);
    const double ilu_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - ilu_t0 : 0.0;
    const double asm_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    MomentumHatData asm_hat = build_candidate(SchurMomentumApproximationType::ASM_K);
    const double asm_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - asm_t0 : 0.0;
    const double score_ilu_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    const double ilu_score = score_candidate(ilu_hat);
    const double score_ilu_t =
        trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - score_ilu_t0 : 0.0;
    const double score_asm_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    const double asm_score = score_candidate(asm_hat);
    const double score_asm_t =
        trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - score_asm_t0 : 0.0;
    if (trace_setup) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] momentum_hat adaptive=1 approx=ilu-k "
                   "build_ilu=%e build_asm=%e score_ilu=%e score_asm=%e "
                   "ilu_score=%e asm_score=%e total=%e\n",
                   ilu_t,
                   asm_t,
                   score_ilu_t,
                   score_asm_t,
                   ilu_score,
                   asm_score,
                   fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
    }
    if (std::isfinite(asm_score) && asm_score < ilu_score) {
      return asm_hat;
    }
    return ilu_hat;
  }

  const double build_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  MomentumHatData candidate = build_candidate(approx);
  if (trace_setup) {
    const char* approx_name = "unknown";
    switch (approx) {
      case SchurMomentumApproximationType::DIAG_K: approx_name = "diag-k"; break;
      case SchurMomentumApproximationType::BLOCKDIAG_K: approx_name = "blockdiag-k"; break;
      case SchurMomentumApproximationType::ILU_K: approx_name = "ilu-k"; break;
      case SchurMomentumApproximationType::ASM_K: approx_name = "asm-k"; break;
    }
    std::fprintf(stderr,
                 "[BICGS_SCHUR_SETUP] momentum_hat grouped=%d approx=%s build=%e total=%e\n",
                 has_grouped_bordered ? 1 : 0,
                 approx_name,
                 fe_fsi_linear_solver::fsils_cpu_t() - build_t0,
                 fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
  }
  return candidate;
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
  const bool trace_setup =
      env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING") && lhs.commu.masF;
  const bool trace_setup_all_ranks =
      env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS");
  const bool trace_setup_emit = trace_setup || trace_setup_all_ranks;
  const double total_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
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
  auto trace_stage = [&](int column_index,
                         const char* stage,
                         const char* phase,
                         double dt = -1.0) {
    if (!trace_setup_emit) {
      return;
    }
    if (dt >= 0.0) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] rank=%d reduced_schur column=%d stage=%s phase=%s dt=%e\n",
                   lhs.commu.task,
                   column_index,
                   stage,
                   phase,
                   dt);
    } else {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] rank=%d reduced_schur column=%d stage=%s phase=%s\n",
                   lhs.commu.task,
                   column_index,
                   stage,
                   phase);
    }
    if (trace_setup_all_ranks) {
      std::fflush(stderr);
    }
  };
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

    const int column_index = static_cast<int>(columns.size());
    const double column_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    const auto& left_entries = projected_left_entries(update);
    const auto& right_entries = projected_right_entries(lhs, update);
    const auto& right_entries_for_gt = projected_right_entries_for_transpose(update);
    if (trace_setup_emit) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] rank=%d reduced_schur column=%d source=update grouped_id=%d "
                   "left_full=%zu right_full=%zu left_owned=%zu right_owned=%zu\n",
                   lhs.commu.task,
                   column_index,
                   update.grouped_coupling_id,
                   left_entries.size(),
                   right_entries.size(),
                   update.left_owned.size(),
                   update.right_owned.size());
      if (trace_setup_all_ranks) {
        std::fflush(stderr);
      }
    }

    ReducedColumn col;
    const double fill_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    trace_stage(column_index, "filled", "begin");
    fill_projected_reduced_vector(lhs, update, left_entries, mom_ncomp, col.momentum_left_hat);
    fill_projected_reduced_vector(lhs, update, right_entries, mom_ncomp, col.momentum_right_owned);
    if (trace_setup_all_ranks && column_index < 2) {
      trace_dense_vector_owned_entries(lhs, "reduced_schur_right_owned", mom_ncomp, col.momentum_right_owned);
      trace_dense_vector_owned_entries(lhs, "reduced_schur_left_before_hat", mom_ncomp, col.momentum_left_hat);
    }
    const double fill_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - fill_t0 : 0.0;
    trace_stage(column_index, "filled", "done", fill_t);

    Array<double> momentum_right_t;
    fill_projected_reduced_vector(lhs, update, right_entries_for_gt, mom_ncomp, momentum_right_t);

    const double hat_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    apply_momentum_hat(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                       mom_ncomp, lhs.nNo, momentum_hat, col.momentum_left_hat);
    const double hat_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - hat_t0 : 0.0;
    trace_stage(column_index, "hat", "done", hat_t);
    const double hat_t_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    apply_momentum_hat_transpose(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                 mom_ncomp, lhs.nNo, momentum_hat, momentum_right_t);
    const double hat_t_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - hat_t_t0 : 0.0;
    trace_stage(column_index, "hat_t", "done", hat_t_t);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    const double sync_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    trace_stage(column_index, "sync_left_hat", "begin");
    halo.sync_owned_to_ghost_vector(mom_ncomp, col.momentum_left_hat, skip_post_solve_halo_sync());
    trace_stage(column_index, "sync_left_hat", "done");
    trace_stage(column_index, "sync_right_hat_t", "begin");
    halo.sync_owned_to_ghost_vector(mom_ncomp, momentum_right_t, skip_post_solve_halo_sync());
    trace_stage(column_index, "sync_right_hat_t", "done");
    if (trace_setup_all_ranks && column_index < 2) {
      trace_dense_vector_owned_entries(lhs, "reduced_schur_left_after_hat", mom_ncomp, col.momentum_left_hat);
      trace_dense_vector_owned_entries(lhs, "reduced_schur_right_after_hat_t", mom_ncomp, momentum_right_t);
    }
    const double sync_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - sync_t0 : 0.0;
    trace_stage(column_index, "sync", "done", sync_t);

    col.schur_left.resize(con_ncomp, lhs.nNo);
    const double d_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    trace_stage(column_index, "d", "begin");
    d_operator.apply(
        dso::ghost_synced_input(mom_ncomp, col.momentum_left_hat),
        dso::owned_only_output(con_ncomp, col.schur_left));
    trace_stage(column_index, "d", "done");
    const double d_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - d_t0 : 0.0;
    const double gt_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    const bool trace_gt_sync =
        env_enabled("SVMP_FSILS_TRACE_REDUCED_SCHUR_GT_SYNC");
    const bool force_gt_sync =
        env_enabled("SVMP_FSILS_REDUCED_SCHUR_FORCE_GT_SYNC");
    if (trace_gt_sync && column_index == 0) {
      const double local_right_l2 =
          norm::fsi_ls_normv(mom_ncomp, lhs.mynNo, lhs.commu, momentum_right_t);
      std::fprintf(stderr,
                   "[BICGS_REDUCED_GT] rank=%d column=%d before_local_gt owned_halo_neighbors=%zu "
                   "nNo=%lld mynNo=%lld con_ncomp=%d mom_ncomp=%d right_l2=%e\n",
                   lhs.commu.task,
                   column_index,
                   lhs.owned_halo_neighbor_ranks.size(),
                   static_cast<long long>(lhs.nNo),
                   static_cast<long long>(lhs.mynNo),
                   con_ncomp,
                   mom_ncomp,
                   local_right_l2);
    }
    trace_stage(column_index, "gt", "begin");
    multiply_rect_transpose_local(lhs.rowPtr, lhs.colPtr, lhs.nNo,
                                  mom_ncomp, con_ncomp, G,
                                  momentum_right_t, col.schur_right);
    if (trace_gt_sync && column_index == 0) {
      const double local_schur_right_l2 =
          norm::fsi_ls_normv(con_ncomp, lhs.mynNo, lhs.commu, col.schur_right);
      std::fprintf(stderr,
                   "[BICGS_REDUCED_GT] rank=%d column=%d after_local_gt "
                   "schur_right_l2=%e force_sync=%d\n",
                   lhs.commu.task,
                   column_index,
                   local_schur_right_l2,
                   force_gt_sync ? 1 : 0);
    }
    // schur_right is the contracting right factor of the low-rank correction
    // and is consumed later only through owned-only dense contractions.
    // Keep it owner-local here; additive overlap is unnecessary and, on some
    // grouped auxiliary paths, can deadlock.
    if (force_gt_sync) {
      trace_stage(column_index, "sync_gt", "begin");
      halo.sync_owned_to_ghost_vector(con_ncomp, col.schur_right, skip_post_solve_halo_sync());
      trace_stage(column_index, "sync_gt", "done");
    }
    if (trace_gt_sync && column_index == 0) {
      const double synced_schur_right_l2 =
          norm::fsi_ls_normv(con_ncomp, lhs.mynNo, lhs.commu, col.schur_right);
      std::fprintf(stderr,
                   "[BICGS_REDUCED_GT] rank=%d column=%d after_sync_gt "
                   "schur_right_l2=%e\n",
                   lhs.commu.task,
                   column_index,
                   synced_schur_right_l2);
    }
    const double gt_t = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - gt_t0 : 0.0;
    trace_stage(column_index, "gt", "done", gt_t);
    columns.push_back(std::move(col));
    if (trace_setup_emit) {
      const double right_owned_l2 =
          dense_dense_owned_norm_local(lhs, mom_ncomp, col.momentum_right_owned);
      const double left_hat_l2 =
          dense_dense_owned_norm_local(lhs, mom_ncomp, col.momentum_left_hat);
      const double schur_left_l2 =
          dense_dense_owned_norm_local(lhs, con_ncomp, col.schur_left);
      const double schur_right_l2 =
          dense_dense_owned_norm_local(lhs, con_ncomp, col.schur_right);
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] rank=%d reduced_schur column=%d fill=%e hat=%e hat_t=%e "
                   "sync=%e d=%e gt=%e |right|=%e |left_hat|=%e |schur_left|=%e |schur_right|=%e total=%e\n",
                   lhs.commu.task,
                   column_index,
                   fill_t,
                   hat_t,
                   hat_t_t,
                   sync_t,
                   d_t,
                   gt_t,
                   right_owned_l2,
                   left_hat_l2,
                   schur_left_l2,
                   schur_right_l2,
                   fe_fsi_linear_solver::fsils_cpu_t() - column_t0);
      if (trace_setup_all_ranks) {
        std::fflush(stderr);
      }
    }
  };

  auto append_face_column = [&](const fe_fsi_linear_solver::FSILS_faceType& face) {
    if (!face.coupledFlag || std::abs(face.res) <= 1e-30) {
      return;
    }

    const int column_index = static_cast<int>(columns.size());
    if (trace_setup_emit) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] rank=%d reduced_schur column=%d source=face face_nNo=%d dof=%d res=%e\n",
                   lhs.commu.task,
                   column_index,
                   face.nNo,
                   face.dof,
                   face.res);
      if (trace_setup_all_ranks) {
        std::fflush(stderr);
      }
    }
    ReducedColumn col;
    trace_stage(column_index, "filled", "begin");
    fill_projected_face_vector(lhs, face, mom_ncomp, col.momentum_left_hat);
    col.momentum_right_owned = col.momentum_left_hat;
    Array<double> momentum_right_t = col.momentum_left_hat;

    apply_momentum_hat(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                       mom_ncomp, lhs.nNo, momentum_hat, col.momentum_left_hat);
    apply_momentum_hat_transpose(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                 mom_ncomp, lhs.nNo, momentum_hat, momentum_right_t);
    const fe_fsi_linear_solver::HaloExchange halo(lhs);
    halo.sync_owned_to_ghost_vector(mom_ncomp, col.momentum_left_hat, skip_post_solve_halo_sync());
    halo.sync_owned_to_ghost_vector(mom_ncomp, momentum_right_t, skip_post_solve_halo_sync());

    col.schur_left.resize(con_ncomp, lhs.nNo);
    d_operator.apply(
        dso::ghost_synced_input(mom_ncomp, col.momentum_left_hat),
        dso::owned_only_output(con_ncomp, col.schur_left));
    multiply_rect_transpose_local(lhs.rowPtr, lhs.colPtr, lhs.nNo,
                                  mom_ncomp, con_ncomp, G,
                                  momentum_right_t, col.schur_right);
    if (env_enabled("SVMP_FSILS_REDUCED_SCHUR_FORCE_GT_SYNC")) {
      halo.sync_owned_to_ghost_vector(con_ncomp, col.schur_right, skip_post_solve_halo_sync());
    }
    columns.push_back(std::move(col));
  };

  std::vector<double> dense_seed;
  dense_seed.reserve(lhs.reduced_updates.size());

  const bool has_grouped_bordered = !lhs.grouped_bordered_field_couplings.empty();
  const double build_columns_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
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
    if (trace_setup_emit) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] rank=%d grouped_group group=%d mode_count=%zu aux_size=%zu\n",
                   lhs.commu.task,
                   group.grouped_coupling_id,
                   group.modes.size(),
                   group.aux_matrix.size());
      if (trace_setup_all_ranks) {
        std::fflush(stderr);
      }
    }
    const int block_rank = static_cast<int>(group.modes.size());
    if (group.aux_matrix.size() != static_cast<std::size_t>(block_rank * block_rank)) {
      continue;
    }

    const int old_rank = static_cast<int>(columns.size());
    for (std::size_t mode_index = 0; mode_index < group.modes.size(); ++mode_index) {
      if (trace_setup_emit) {
        std::fprintf(stderr,
                     "[BICGS_SCHUR_SETUP] rank=%d grouped_begin group=%d mode=%zu old_rank=%d\n",
                     lhs.commu.task,
                     group.grouped_coupling_id,
                     mode_index,
                     old_rank);
      }
      const auto& mode = group.modes[mode_index];
      append_column(mode);
      if (trace_setup_emit) {
        std::fprintf(stderr,
                     "[BICGS_SCHUR_SETUP] rank=%d grouped_end group=%d mode=%zu rank_now=%zu\n",
                     lhs.commu.task,
                     group.grouped_coupling_id,
                     mode_index,
                     columns.size());
      }
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
  const double build_columns_t =
      trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - build_columns_t0 : 0.0;

  const int rank = static_cast<int>(columns.size());
  if (rank == 0 || dense_seed.size() != static_cast<std::size_t>(rank * rank)) {
    if (trace_setup) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] reduced_schur rank=0 early_exit=1 build_columns=%e total=%e\n",
                   build_columns_t,
                   fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
    }
    return;
  }

  const double dense_m_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  std::vector<double> dense_m = dense_seed;
  std::vector<double> dense_m_local(static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    for (int j = 0; j < rank; ++j) {
      const double local_dot =
          dense_dense_owned_dot_local(lhs, mom_ncomp,
                                      columns[static_cast<std::size_t>(i)].momentum_right_owned,
                                      columns[static_cast<std::size_t>(j)].momentum_left_hat);
      dense_m_local[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                    static_cast<std::size_t>(j)] = local_dot;
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
              static_cast<std::size_t>(j)] += local_dot;
    }
  }
  if (trace_setup_emit && env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS")) {
    trace_dense_matrix(lhs, "reduced_schur_dense_m_local", dense_m_local, rank);
  }
  allreduce_sum_in_place(lhs, dense_m);
  trace_dense_matrix(lhs, "reduced_schur_dense_m", dense_m, rank);
  const double dense_m_t =
      trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - dense_m_t0 : 0.0;

  pc.low_rank_right.clear();
  pc.low_rank_preconditioned_left.clear();
  pc.low_rank_right.reserve(columns.size());
  pc.low_rank_preconditioned_left.reserve(columns.size());
  const double precondition_left_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  for (std::size_t column_index = 0; column_index < columns.size(); ++column_index) {
    auto& column = columns[column_index];
    pc.low_rank_right.push_back(column.schur_right);
    Array<double> z = column.schur_left;
    apply_base_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr,
                                    pc, con_ncomp, lhs.nNo, z);
    sync_schur_halo(lhs, con_ncomp, z);
    pc.low_rank_preconditioned_left.push_back(std::move(z));
  }
  const double precondition_left_t =
      trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - precondition_left_t0 : 0.0;

  const double dense_ctz_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  std::vector<double> dense_ctz(static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank), 0.0);
  std::vector<double> dense_ctz_local(static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    for (int j = 0; j < rank; ++j) {
      const double local_dot =
          dense_dense_owned_dot_local(lhs, con_ncomp,
                                      pc.low_rank_right[static_cast<std::size_t>(i)],
                                      pc.low_rank_preconditioned_left[static_cast<std::size_t>(j)]);
      dense_ctz_local[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                      static_cast<std::size_t>(j)] = local_dot;
      dense_ctz[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                static_cast<std::size_t>(j)] = local_dot;
    }
  }
  if (trace_setup_emit && env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS")) {
    trace_dense_matrix(lhs, "reduced_schur_dense_ctz_local", dense_ctz_local, rank);
  }
  allreduce_sum_in_place(lhs, dense_ctz);
  trace_dense_matrix(lhs, "reduced_schur_dense_ctz", dense_ctz, rank);
  const double dense_ctz_t =
      trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() - dense_ctz_t0 : 0.0;

  for (std::size_t idx = 0; idx < dense_m.size(); ++idx) {
    dense_m[idx] += dense_ctz[idx];
  }
  trace_dense_matrix(lhs, "reduced_schur_dense_total", dense_m, rank);

  const double invert_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  pc.low_rank_inner_inv.assign(dense_m.size(), 0.0);
  if (!invert_dense_block(rank, dense_m.data(), pc.low_rank_inner_inv.data())) {
    pc.low_rank_right.clear();
    pc.low_rank_preconditioned_left.clear();
    pc.low_rank_inner_inv.clear();
  }
  if (trace_setup) {
    const double invert_t = fe_fsi_linear_solver::fsils_cpu_t() - invert_t0;
    std::fprintf(stderr,
                 "[BICGS_SCHUR_SETUP] reduced_schur rank=%d build_columns=%e dense_m=%e "
                 "precondition_left=%e dense_ctz=%e invert=%e total=%e\n",
                 rank,
                 build_columns_t,
                 dense_m_t,
                 precondition_left_t,
                 dense_ctz_t,
                 invert_t,
                 fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
  }
}

void apply_dense_low_rank_correction(
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    int con_ncomp,
    fsils_int nNo,
    const std::vector<Array<double>>& right,
    const std::vector<Array<double>>& preconditioned_left,
    const std::vector<double>& inner_inv,
    Array<double>& x)
{
  const int rank = static_cast<int>(right.size());
  if (rank == 0 || preconditioned_left.size() != right.size() ||
      inner_inv.size() != static_cast<std::size_t>(rank * rank)) {
    return;
  }

  std::vector<double> gamma(static_cast<std::size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    gamma[static_cast<std::size_t>(i)] =
        dense_dense_owned_dot_local(lhs, con_ncomp, right[static_cast<std::size_t>(i)], x);
  }
  allreduce_sum_in_place(lhs, gamma);

  std::vector<double> delta(static_cast<std::size_t>(rank), 0.0);
  for (int i = 0; i < rank; ++i) {
    double sum = 0.0;
    for (int j = 0; j < rank; ++j) {
      sum += inner_inv[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                       static_cast<std::size_t>(j)] *
             gamma[static_cast<std::size_t>(j)];
    }
    delta[static_cast<std::size_t>(i)] = sum;
  }

  for (int i = 0; i < rank; ++i) {
    const double scale = delta[static_cast<std::size_t>(i)];
    if (std::abs(scale) <= 1e-30) {
      continue;
    }
    const auto& z = preconditioned_left[static_cast<std::size_t>(i)];
    for (fsils_int node = 0; node < nNo; ++node) {
      for (int comp = 0; comp < con_ncomp; ++comp) {
        x(comp, node) -= z(comp, node) * scale;
      }
    }
  }
}

void build_partition_schur_coarse_correction(
    const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr,
    const Vector<fsils_int>& diagPtr,
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    int con_ncomp,
    SchurPreconditionerData& pc)
{
  pc.coarse_right.clear();
  pc.coarse_preconditioned_left.clear();
  pc.coarse_inner_inv.clear();

  if (!schur_partition_coarse_modes_enabled() || lhs.commu.nTasks <= 1 || con_ncomp != 1) {
    return;
  }
  if (pc.momentum_hat.point_inv.nrows() == 0 || pc.momentum_hat.point_inv.ncols() == 0) {
    return;
  }

  std::vector<double> owned_counts(static_cast<std::size_t>(lhs.commu.nTasks), 0.0);
  const double local_owned = static_cast<double>(lhs.mynNo);
  MPI_Allgather(&local_owned,
                1,
                cm_mod::mpreal,
                owned_counts.data(),
                1,
                cm_mod::mpreal,
                lhs.commu.comm);

  std::vector<int> active_ranks;
  active_ranks.reserve(static_cast<std::size_t>(lhs.commu.nTasks));
  for (int rank = 0; rank < lhs.commu.nTasks; ++rank) {
    if (owned_counts[static_cast<std::size_t>(rank)] > 0.5) {
      active_ranks.push_back(rank);
    }
  }
  if (active_ranks.size() < 2) {
    return;
  }

  const int reference_rank = active_ranks.back();
  const double reference_count = owned_counts[static_cast<std::size_t>(reference_rank)];
  if (!(reference_count > 0.5)) {
    return;
  }

  pc.coarse_right.reserve(active_ranks.size() - 1);
  pc.coarse_preconditioned_left.reserve(active_ranks.size() - 1);

  for (const int coarse_rank : active_ranks) {
    if (coarse_rank == reference_rank) {
      continue;
    }

    Array<double> mode(con_ncomp, lhs.nNo);
    mode = 0.0;
    if (lhs.commu.task == coarse_rank) {
      for (fsils_int node = 0; node < lhs.mynNo; ++node) {
        mode(0, node) = 1.0;
      }
    } else if (lhs.commu.task == reference_rank) {
      const double weight =
          -owned_counts[static_cast<std::size_t>(coarse_rank)] / reference_count;
      for (fsils_int node = 0; node < lhs.mynNo; ++node) {
        mode(0, node) = weight;
      }
    }

    Array<double> z;
    apply_hat_schur_operator(rowPtr, colPtr, diagPtr, lhs, pc, mode, z);
    apply_base_schur_preconditioner(rowPtr, colPtr, diagPtr, pc, con_ncomp, lhs.nNo, z);
    sync_schur_halo(lhs, con_ncomp, z);
    apply_dense_low_rank_correction(lhs,
                                    con_ncomp,
                                    lhs.nNo,
                                    pc.low_rank_right,
                                    pc.low_rank_preconditioned_left,
                                    pc.low_rank_inner_inv,
                                    z);

    pc.coarse_right.push_back(std::move(mode));
    pc.coarse_preconditioned_left.push_back(std::move(z));
  }

  const int coarse_rank_count = static_cast<int>(pc.coarse_right.size());
  if (coarse_rank_count == 0) {
    return;
  }

  std::vector<double> dense_m(static_cast<std::size_t>(coarse_rank_count) *
                                  static_cast<std::size_t>(coarse_rank_count),
                              0.0);
  for (int i = 0; i < coarse_rank_count; ++i) {
    for (int j = 0; j < coarse_rank_count; ++j) {
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(coarse_rank_count) +
              static_cast<std::size_t>(j)] =
          dense_dense_owned_dot_local(
              lhs,
              con_ncomp,
              pc.coarse_right[static_cast<std::size_t>(i)],
              pc.coarse_preconditioned_left[static_cast<std::size_t>(j)]);
    }
  }
  allreduce_sum_in_place(lhs, dense_m);

  pc.coarse_inner_inv.assign(dense_m.size(), 0.0);
  if (!invert_dense_block(coarse_rank_count, dense_m.data(), pc.coarse_inner_inv.data())) {
    pc.coarse_right.clear();
    pc.coarse_preconditioned_left.clear();
    pc.coarse_inner_inv.clear();
  }
}

void build_global_mean_schur_coarse_correction(
    const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr,
    const Vector<fsils_int>& diagPtr,
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    int con_ncomp,
    SchurPreconditionerData& pc)
{
  if (!schur_global_mean_coarse_mode_enabled() || con_ncomp != 1) {
    return;
  }
  if (pc.momentum_hat.point_inv.nrows() == 0 || pc.momentum_hat.point_inv.ncols() == 0) {
    return;
  }

  Array<double> mode(con_ncomp, lhs.nNo);
  mode = 0.0;
  for (fsils_int node = 0; node < lhs.mynNo; ++node) {
    mode(0, node) = 1.0;
  }

  Array<double> z;
  apply_hat_schur_operator(rowPtr, colPtr, diagPtr, lhs, pc, mode, z);
  apply_base_schur_preconditioner(rowPtr, colPtr, diagPtr, pc, con_ncomp, lhs.nNo, z);
  sync_schur_halo(lhs, con_ncomp, z);
  apply_dense_low_rank_correction(lhs,
                                  con_ncomp,
                                  lhs.nNo,
                                  pc.low_rank_right,
                                  pc.low_rank_preconditioned_left,
                                  pc.low_rank_inner_inv,
                                  z);

  const std::size_t old_count = pc.coarse_right.size();
  pc.coarse_right.push_back(std::move(mode));
  pc.coarse_preconditioned_left.push_back(std::move(z));

  const int coarse_rank_count = static_cast<int>(pc.coarse_right.size());
  std::vector<double> dense_m(static_cast<std::size_t>(coarse_rank_count) *
                                  static_cast<std::size_t>(coarse_rank_count),
                              0.0);
  for (int i = 0; i < coarse_rank_count; ++i) {
    for (int j = 0; j < coarse_rank_count; ++j) {
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(coarse_rank_count) +
              static_cast<std::size_t>(j)] =
          dense_dense_owned_dot_local(
              lhs,
              con_ncomp,
              pc.coarse_right[static_cast<std::size_t>(i)],
              pc.coarse_preconditioned_left[static_cast<std::size_t>(j)]);
    }
  }
  allreduce_sum_in_place(lhs, dense_m);

  pc.coarse_inner_inv.assign(dense_m.size(), 0.0);
  if (!invert_dense_block(coarse_rank_count, dense_m.data(), pc.coarse_inner_inv.data())) {
    pc.coarse_right.resize(old_count);
    pc.coarse_preconditioned_left.resize(old_count);
    if (old_count == 0u) {
      pc.coarse_inner_inv.clear();
    } else {
      std::vector<double> fallback_dense(static_cast<std::size_t>(old_count) *
                                             static_cast<std::size_t>(old_count),
                                         0.0);
      for (std::size_t i = 0; i < old_count; ++i) {
        for (std::size_t j = 0; j < old_count; ++j) {
          fallback_dense[i * old_count + j] =
              dense_dense_owned_dot_local(lhs,
                                          con_ncomp,
                                          pc.coarse_right[i],
                                          pc.coarse_preconditioned_left[j]);
        }
      }
      allreduce_sum_in_place(lhs, fallback_dense);
      pc.coarse_inner_inv.assign(fallback_dense.size(), 0.0);
      if (!invert_dense_block(static_cast<int>(old_count),
                              fallback_dense.data(),
                              pc.coarse_inner_inv.data())) {
        pc.coarse_right.clear();
        pc.coarse_preconditioned_left.clear();
        pc.coarse_inner_inv.clear();
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

  const bool trace_setup =
      env_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING") && lhs.commu.masF;
  const double total_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  SchurPreconditionerData pc{};
  const int forced_preconditioner = forced_schur_preconditioner_code();
  const auto preconditioner =
      (forced_preconditioner >= 0)
          ? static_cast<SchurPreconditionerType>(forced_preconditioner)
          : ls.schur_preconditioner;
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
  build_point_inverse_blocks(lhs.diagPtr, &lhs, con_ncomp, lhs.nNo, L, diagonal_only, pc.point_inv);
  pc.operator_D = &D;
  pc.operator_G = &G;
  pc.operator_L_storage = L;
  pc.operator_L = &pc.operator_L_storage;
  pc.operator_mom_ncomp = mom_ncomp;
  pc.operator_con_ncomp = con_ncomp;
  if (!lhs.reduced_updates.empty() || !lhs.grouped_bordered_field_couplings.empty()) {
    // The OOP BlockSchur path permutes the per-node ordering to the explicit
    // momentum/constraint block layout before entering FSILS. The Schur
    // subsolver does not carry the original block starts, so project the exact
    // outlet corrections using the FSILS-local block order.
    build_explicit_schur_corrections(lhs, /*mom_start=*/0, mom_ncomp,
                                     /*con_start=*/mom_ncomp, con_ncomp, pc);
  }

  if (need_momentum_hat) {
    const double hat_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    pc.momentum_hat =
        build_momentum_hat_data(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, lhs.nNo,
                                mom_ncomp, K, ls.schur_momentum_approximation);
    if (trace_setup) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] preconditioner momentum_hat=%e\n",
                   fe_fsi_linear_solver::fsils_cpu_t() - hat_t0);
    }
  }

  if (preconditioner == SchurPreconditionerType::ILU_L) {
    factorize_block_ilu0(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs.nNo, con_ncomp, L,
                         pc.ilu_factors, pc.ilu_diag_inv);
    pc.use_ilu = true;
    if (has_exact_schur_corrections) {
      const double reduced_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
      build_reduced_schur_correction(lhs, mom_ncomp, con_ncomp, D, G, pc.momentum_hat, pc);
      if (trace_setup) {
        std::fprintf(stderr,
                     "[BICGS_SCHUR_SETUP] preconditioner reduced_schur=%e\n",
                     fe_fsi_linear_solver::fsils_cpu_t() - reduced_t0);
      }
    }
    const double coarse_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    build_partition_schur_coarse_correction(
        lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, con_ncomp, pc);
    build_global_mean_schur_coarse_correction(
        lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, con_ncomp, pc);
    if (trace_setup) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] preconditioner coarse=%e total=%e\n",
                   fe_fsi_linear_solver::fsils_cpu_t() - coarse_t0,
                   fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
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
    pc.operator_refinement_steps = cfg.operator_refinement_steps;
    pc.operator_refinement_omega = cfg.operator_refinement_omega;
  }

  if (has_exact_schur_corrections) {
    const double reduced_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
    build_reduced_schur_correction(lhs, mom_ncomp, con_ncomp, D, G, pc.momentum_hat, pc);
    if (trace_setup) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_SETUP] preconditioner reduced_schur=%e\n",
                   fe_fsi_linear_solver::fsils_cpu_t() - reduced_t0);
    }
  }
  const double coarse_t0 = trace_setup ? fe_fsi_linear_solver::fsils_cpu_t() : 0.0;
  build_partition_schur_coarse_correction(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, con_ncomp, pc);
  build_global_mean_schur_coarse_correction(
      lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, con_ncomp, pc);
  if (trace_setup) {
    std::fprintf(stderr,
                 "[BICGS_SCHUR_SETUP] preconditioner coarse=%e total=%e\n",
                 fe_fsi_linear_solver::fsils_cpu_t() - coarse_t0,
                 fe_fsi_linear_solver::fsils_cpu_t() - total_t0);
  }

  return pc;
}

void apply_schur_low_rank_correction(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                     const SchurPreconditionerData& pc,
                                     int con_ncomp,
                                     fsils_int nNo,
                                     Array<double>& x)
{
  apply_dense_low_rank_correction(lhs,
                                  con_ncomp,
                                  nNo,
                                  pc.low_rank_right,
                                  pc.low_rank_preconditioned_left,
                                  pc.low_rank_inner_inv,
                                  x);
}

void apply_schur_partition_coarse_correction(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                             const SchurPreconditionerData& pc,
                                             int con_ncomp,
                                             fsils_int nNo,
                                             Array<double>& x)
{
  apply_dense_low_rank_correction(lhs,
                                  con_ncomp,
                                  nNo,
                                  pc.coarse_right,
                                  pc.coarse_preconditioned_left,
                                  pc.coarse_inner_inv,
                                  x);
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
  const fe_fsi_linear_solver::HaloExchange hat_halo(lhs);
  schur_ops.GL.apply(
      dso::ghost_synced_input(con_ncomp, in_vec),
      dso::owned_only_output(mom_ncomp, pc.scratch_gp),
      dso::owned_only_output(con_ncomp, pc.scratch_sp));
  apply_explicit_constraint_schur_corrections(lhs, pc, in_vec, pc.scratch_gp, pc.scratch_sp);
  hat_halo.sync_owned_to_ghost_vector(mom_ncomp, pc.scratch_gp, skip_post_solve_halo_sync());

  pc.scratch_hgp = pc.scratch_gp;
  apply_momentum_hat(rowPtr, colPtr, diagPtr, mom_ncomp, nNo, pc.momentum_hat, pc.scratch_hgp);
  hat_halo.sync_owned_to_ghost_vector(mom_ncomp, pc.scratch_hgp, skip_post_solve_halo_sync());

  schur_ops.D.apply(
      dso::ghost_synced_input(mom_ncomp, pc.scratch_hgp),
      dso::owned_only_output(con_ncomp, pc.scratch_dgp));
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
  sync_schur_halo(lhs, con_ncomp, x);
  apply_schur_low_rank_correction(lhs, pc, con_ncomp, nNo, x);
  apply_schur_partition_coarse_correction(lhs, pc, con_ncomp, nNo, x);

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
    sync_schur_halo(lhs, con_ncomp, pc.scratch_correction);
    apply_schur_low_rank_correction(lhs, pc, con_ncomp, nNo, pc.scratch_correction);
    apply_schur_partition_coarse_correction(lhs, pc, con_ncomp, nNo, pc.scratch_correction);

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
  Vector<double> M_diag(nNo);
  Vector<double> M_inv(nNo);
  Vector<double> in_sync(nNo);
  static std::uint64_t legacy_schur_dump_counter = 0;
  const char* legacy_dump_prefix = schur_dump_prefix();
  const std::uint64_t legacy_dump_solve_id =
      (legacy_dump_prefix == nullptr) ? 0 : legacy_schur_dump_counter++;
  const int legacy_dump_max_solves = schur_dump_max_solves();
  const int legacy_dump_max_iters = schur_dump_max_iters();
  const bool legacy_dump_enabled =
      legacy_dump_prefix != nullptr &&
      (legacy_dump_max_solves < 0 ||
       legacy_dump_solve_id < static_cast<std::uint64_t>(legacy_dump_max_solves));
  auto dump_legacy = [&](const char* label, const Vector<double>& values) {
    if (legacy_dump_enabled) {
      dump_schur_vector_owned(lhs, legacy_dump_prefix, legacy_dump_solve_id, label, values);
    }
  };

  const Vector<double> rhs_unpreconditioned = R;
  dump_legacy("rhs_unpreconditioned", rhs_unpreconditioned);
  for (fsils_int i = 0; i < nNo; ++i) {
    M_diag(i) = L_values(lhs.diagPtr(i));
  }
  fe_fsi_linear_solver::HaloExchange(lhs).sync_owned_to_ghost_scalar(M_diag);
  for (fsils_int i = 0; i < nNo; ++i) {
    const double diag_val = M_diag(i);
    M_inv(i) = (std::abs(diag_val) > 1e-12) ? 1.0 / diag_val : 1.0;
  }

  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; ++i) {
    R(i) *= M_inv(i);
  }
  const Vector<double> rhs_reference = R;
  dump_legacy("diagonal_L", M_diag);
  dump_legacy("diagonal_L_inv", M_inv);
  dump_legacy("rhs_preconditioned_Minv", rhs_reference);

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
    if (halo.has_owned_halo() && !skip_post_solve_halo_sync()) {
      // BiCGStab updates overlap vectors locally; synchronize before reuse in the
      // next Schur matvec so owned values drive ghost entries consistently.
      in_sync = in_vec;
      halo.sync_owned_to_ghost_scalar(in_sync);
      op_in = &in_sync;
    }

    GL.apply(
        dso::ghost_synced_input(*op_in),
        dso::owned_only_output(nsd, GP),
        dso::owned_only_output(SP));
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, nsd, GP, GP);
    if (halo.has_owned_halo() && !skip_post_solve_halo_sync()) {
      // The coupled-face correction modifies a momentum overlap field in place.
      // Synchronize before reusing it in D*(...) on neighboring owned rows.
      halo.sync_owned_to_ghost_vector(nsd, GP);
    }
    D.apply(
        dso::ghost_synced_input(nsd, GP),
        dso::owned_only_output(DGP));

    #pragma omp parallel for schedule(static)
    for (fsils_int i = 0; i < nNo; ++i) {
      out_vec(i) = M_inv(i) * (SP(i) - DGP(i));
    }

    halo.sync_owned_to_ghost_scalar(out_vec, skip_post_solve_halo_sync());
  };

  auto dump_legacy_operator_decomposition = [&](const char* label, const Vector<double>& in_vec) {
    if (!legacy_dump_enabled) {
      return;
    }

    const Vector<double>* op_in = &in_vec;
    Vector<double> probe_in_sync(nNo);
    const fe_fsi_linear_solver::HaloExchange probe_halo(lhs);
    if (probe_halo.has_owned_halo() && !skip_post_solve_halo_sync()) {
      probe_in_sync = in_vec;
      probe_halo.sync_owned_to_ghost_scalar(probe_in_sync);
      op_in = &probe_in_sync;
    }

    Array<double> probe_gp(nsd, nNo);
    Vector<double> probe_sp(nNo), probe_dgp(nNo), probe_raw(nNo), probe_out(nNo);
    GL.apply(
        dso::ghost_synced_input(*op_in),
        dso::owned_only_output(nsd, probe_gp),
        dso::owned_only_output(probe_sp));
    dump_schur_array_owned(
        lhs,
        legacy_dump_prefix,
        legacy_dump_solve_id,
        (std::string("legacy.operator_") + label + "_GP_after_GL").c_str(),
        probe_gp);
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, nsd, probe_gp, probe_gp);
    dump_schur_array_owned(
        lhs,
        legacy_dump_prefix,
        legacy_dump_solve_id,
        (std::string("legacy.operator_") + label + "_GP_after_add_bc").c_str(),
        probe_gp);
    if (probe_halo.has_owned_halo() && !skip_post_solve_halo_sync()) {
      probe_halo.sync_owned_to_ghost_vector(nsd, probe_gp);
    }
    dump_schur_array_owned(
        lhs,
        legacy_dump_prefix,
        legacy_dump_solve_id,
        (std::string("legacy.operator_") + label + "_GP_before_D").c_str(),
        probe_gp);
    D.apply(
        dso::ghost_synced_input(nsd, probe_gp),
        dso::owned_only_output(probe_dgp));
    #pragma omp parallel for schedule(static)
    for (fsils_int i = 0; i < nNo; ++i) {
      probe_raw(i) = probe_sp(i) - probe_dgp(i);
      probe_out(i) = M_inv(i) * probe_raw(i);
    }
    probe_halo.sync_owned_to_ghost_scalar(probe_out, skip_post_solve_halo_sync());

    dump_legacy((std::string("operator_") + label + "_input").c_str(), in_vec);
    dump_legacy((std::string("operator_") + label + "_SP").c_str(), probe_sp);
    dump_legacy((std::string("operator_") + label + "_DGP").c_str(), probe_dgp);
    dump_legacy((std::string("operator_") + label + "_raw_SP_minus_DGP").c_str(), probe_raw);
    dump_legacy((std::string("operator_") + label + "_Minv_raw").c_str(), probe_out);
  };

  if (legacy_dump_enabled) {
    dump_legacy_operator_decomposition("rhs", rhs_reference);

    Vector<double> global_mean_probe(nNo);
    global_mean_probe = 0.0;
    for (fsils_int node = 0; node < lhs.mynNo; ++node) {
      global_mean_probe(node) = 1.0;
    }
    dump_legacy_operator_decomposition("global_mean", global_mean_probe);

#if FE_HAS_MPI
    if (lhs.commu.nTasks > 1) {
      std::vector<double> owned_counts(static_cast<std::size_t>(lhs.commu.nTasks), 0.0);
      const double local_owned = static_cast<double>(lhs.mynNo);
      MPI_Allgather(&local_owned,
                    1,
                    cm_mod::mpreal,
                    owned_counts.data(),
                    1,
                    cm_mod::mpreal,
                    lhs.commu.comm);

      std::vector<int> active_ranks;
      active_ranks.reserve(static_cast<std::size_t>(lhs.commu.nTasks));
      for (int rank = 0; rank < lhs.commu.nTasks; ++rank) {
        if (owned_counts[static_cast<std::size_t>(rank)] > 0.5) {
          active_ranks.push_back(rank);
        }
      }
      if (active_ranks.size() >= 2u) {
        const int reference_rank = active_ranks.back();
        const int probe_rank = active_ranks.front();
        const double reference_count = owned_counts[static_cast<std::size_t>(reference_rank)];
        if (reference_count > 0.5 && reference_rank != probe_rank) {
          Vector<double> partition_probe(nNo);
          partition_probe = 0.0;
          if (lhs.commu.task == probe_rank) {
            for (fsils_int node = 0; node < lhs.mynNo; ++node) {
              partition_probe(node) = 1.0;
            }
          } else if (lhs.commu.task == reference_rank) {
            const double weight =
                -owned_counts[static_cast<std::size_t>(probe_rank)] / reference_count;
            for (fsils_int node = 0; node < lhs.mynNo; ++node) {
              partition_probe(node) = weight;
            }
          }
          dump_legacy_operator_decomposition("partition_0", partition_probe);
        }
      }
    }
#endif
  }

  int active_coupled_faces = static_cast<int>(std::count_if(
      lhs.face.begin(), lhs.face.end(),
      [](const auto& face) {
        return face.coupledFlag && std::abs(face.res) > 1e-30 && face.nNo > 0;
      }));
  const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
  auto compute_scalar_stats = [&](const Vector<double>& vec) {
    struct ScalarStats {
      double l2{0.0};
      double mean{0.0};
      double centered_l2{0.0};
      double count{0.0};
      double min{0.0};
      double max{0.0};
    };

    const fsils_int local_n = std::min<fsils_int>(mynNo, vec.size());
    double local_sq_sum = 0.0;
    double local_sum = 0.0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    for (fsils_int node = 0; node < local_n; ++node) {
      const double value = vec(node);
      local_sq_sum += value * value;
      local_sum += value;
      local_min = std::min(local_min, value);
      local_max = std::max(local_max, value);
    }

    double global_sq_sum = local_sq_sum;
    double global_sum = local_sum;
    double global_count = static_cast<double>(local_n);
    fe_fsi_linear_solver::fsils_allreduce_sum_in_place(&global_sq_sum, 1, cm_mod::mpreal, lhs.commu);
    fe_fsi_linear_solver::fsils_allreduce_sum_in_place(&global_sum, 1, cm_mod::mpreal, lhs.commu);
    fe_fsi_linear_solver::fsils_allreduce_sum_in_place(&global_count, 1, cm_mod::mpreal, lhs.commu);

    double global_min = local_min;
    double global_max = local_max;
    if (global_count > 0.0) {
      fe_fsi_linear_solver::fsils_allreduce_in_place(&global_min, 1, cm_mod::mpreal, MPI_MIN, lhs.commu);
      fe_fsi_linear_solver::fsils_allreduce_in_place(&global_max, 1, cm_mod::mpreal, MPI_MAX, lhs.commu);
    } else {
      global_min = 0.0;
      global_max = 0.0;
    }

    ScalarStats stats;
    stats.l2 = std::sqrt(std::max(0.0, global_sq_sum));
    stats.mean = (global_count > 0.0) ? (global_sum / global_count) : 0.0;
    stats.count = global_count;
    const double centered_sq =
        std::max(0.0, global_sq_sum - global_count * stats.mean * stats.mean);
    stats.centered_l2 = std::sqrt(centered_sq);
    stats.min = global_min;
    stats.max = global_max;
    return stats;
  };
  auto compute_vector_stats = [&](const Array<double>& vec) {
    struct VectorStats {
      double l2{0.0};
      double max_abs{0.0};
    };

    VectorStats stats;
    stats.l2 = norm::fsi_ls_normv(nsd, mynNo, lhs.commu, vec);
    double local_max_abs = 0.0;
    for (fsils_int node = 0; node < mynNo; ++node) {
      for (int comp = 0; comp < nsd; ++comp) {
        local_max_abs = std::max(local_max_abs, std::abs(vec(comp, node)));
      }
    }
    stats.max_abs = local_max_abs;
    fe_fsi_linear_solver::fsils_allreduce_in_place(&stats.max_abs, 1, cm_mod::mpreal, MPI_MAX, lhs.commu);
    return stats;
  };
  {
    int global_active_coupled_faces = active_coupled_faces;
    collectives.allreduce_sum(active_coupled_faces, global_active_coupled_faces);
    active_coupled_faces = global_active_coupled_faces;
  }
  const bool use_multi_face_gmres =
      active_coupled_faces > 1 &&
      (std::getenv("SVMP_FSILS_DISABLE_MULTI_FACE_LEGACY_GMRES") == nullptr);

  static bool emitted_schur_operator_probe = false;
  if (!emitted_schur_operator_probe && env_enabled("SVMP_FSILS_TRACE_SCHUR_OPERATOR_PROBE")) {
    emitted_schur_operator_probe = true;
    const fe_fsi_linear_solver::HaloExchange probe_halo(lhs);
    Vector<double> probe_in = R;
    Vector<double> probe_in_sync(nNo);
    const Vector<double>* op_in = &probe_in;
    if (probe_halo.has_owned_halo() && !skip_post_solve_halo_sync()) {
      probe_in_sync = probe_in;
      probe_halo.sync_owned_to_ghost_scalar(probe_in_sync);
      op_in = &probe_in_sync;
    }

    Array<double> probe_gp(nsd, nNo);
    Array<double> probe_gp_pre(nsd, nNo);
    Vector<double> probe_sp(nNo);
    Vector<double> probe_dgp(nNo);
    Vector<double> probe_out(nNo);

    GL.apply(
        dso::ghost_synced_input(*op_in),
        dso::owned_only_output(nsd, probe_gp),
        dso::owned_only_output(probe_sp));
    probe_gp_pre = probe_gp;
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, nsd, probe_gp, probe_gp);
    if (probe_halo.has_owned_halo() && !skip_post_solve_halo_sync()) {
      probe_halo.sync_owned_to_ghost_vector(nsd, probe_gp);
    }
    D.apply(
        dso::ghost_synced_input(nsd, probe_gp),
        dso::owned_only_output(probe_dgp));
    #pragma omp parallel for schedule(static)
    for (fsils_int i = 0; i < nNo; ++i) {
      probe_out(i) = M_inv(i) * (probe_sp(i) - probe_dgp(i));
    }
    probe_halo.sync_owned_to_ghost_scalar(probe_out, skip_post_solve_halo_sync());

    const auto in_stats = compute_scalar_stats(probe_in);
    const auto sp_stats = compute_scalar_stats(probe_sp);
    const auto dgp_stats = compute_scalar_stats(probe_dgp);
    const auto out_stats = compute_scalar_stats(probe_out);
    const auto gp_pre_stats = compute_vector_stats(probe_gp_pre);
    const auto gp_post_stats = compute_vector_stats(probe_gp);

    if (lhs.commu.masF) {
      std::fprintf(stderr,
                   "[BICGS_SCHUR_PROBE] distributed=%d active_faces=%d use_multi_face_gmres=%d\n",
                   collectives.distributed() ? 1 : 0,
                   active_coupled_faces,
                   use_multi_face_gmres ? 1 : 0);
      std::fprintf(stderr,
                   "[BICGS_SCHUR_PROBE] input l2=%e mean=%e min=%e max=%e\n",
                   in_stats.l2,
                   in_stats.mean,
                   in_stats.min,
                   in_stats.max);
      std::fprintf(stderr,
                   "[BICGS_SCHUR_PROBE] GL_sp l2=%e mean=%e min=%e max=%e\n",
                   sp_stats.l2,
                   sp_stats.mean,
                   sp_stats.min,
                   sp_stats.max);
      std::fprintf(stderr,
                   "[BICGS_SCHUR_PROBE] GL_gp_pre_bc l2=%e max_abs=%e\n",
                   gp_pre_stats.l2,
                   gp_pre_stats.max_abs);
      std::fprintf(stderr,
                   "[BICGS_SCHUR_PROBE] GL_gp_post_bc l2=%e max_abs=%e\n",
                   gp_post_stats.l2,
                   gp_post_stats.max_abs);
      std::fprintf(stderr,
                   "[BICGS_SCHUR_PROBE] D_gp l2=%e mean=%e min=%e max=%e\n",
                   dgp_stats.l2,
                   dgp_stats.mean,
                   dgp_stats.min,
                   dgp_stats.max);
      std::fprintf(stderr,
                   "[BICGS_SCHUR_PROBE] output l2=%e mean=%e min=%e max=%e\n",
                   out_stats.l2,
                   out_stats.mean,
                   out_stats.min,
                   out_stats.max);
      std::fflush(stderr);
    }
  }

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
    Vector<double> constant_mode(nNo);
    constant_mode = 0.0;
    Vector<double> inv_minv_mode(nNo);
    inv_minv_mode = 0.0;
    for (fsils_int node = 0; node < lhs.mynNo; ++node) {
      constant_mode(node) = 1.0;
      const double minv = M_inv(node);
      inv_minv_mode(node) = (std::abs(minv) > 1e-30) ? (1.0 / minv) : 0.0;
    }
    // These norms no longer feed any result. They are kept so that the
    // allreduce sequence of the solve and the per-solve allreduce count
    // printed by fsils_solve stay unchanged.
    static_cast<void>(norm::fsi_ls_norms(mynNo, lhs.commu, constant_mode));
    static_cast<void>(norm::fsi_ls_norms(mynNo, lhs.commu, M_inv));
    static_cast<void>(norm::fsi_ls_norms(mynNo, lhs.commu, inv_minv_mode));
    ls.callD = fsils_cpu_t();
    ls.suc = false;
    ls.itr = 0;
    const bool use_preconditioned_rhs_initial_guess =
        env_enabled("SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS");
    if (use_preconditioned_rhs_initial_guess) {
      Xg = R;
    } else {
      Xg = 0.0;
    }
    dump_legacy("gmres_initial_guess", Xg);

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
      if (legacy_dump_enabled && total_itr == 0) {
        dump_legacy("gmres_initial_operator", ax);
        dump_legacy("gmres_initial_residual", residual);
      }
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
        if (legacy_dump_enabled && restart_cycles == 1 && i < legacy_dump_max_iters) {
          dump_legacy((std::string("gmres_iter") + std::to_string(i) + "_basis").c_str(), ui);
          dump_legacy((std::string("gmres_iter") + std::to_string(i) + "_operator").c_str(), uip1);
        }

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
        const double abs_diag = std::abs(h(j, j));
        if (abs_diag > std::numeric_limits<double>::epsilon()) {
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

    // This norm no longer feeds any result. It is kept, like the norms before
    // the restart loop, so that the allreduce sequence of the solve and the
    // per-solve allreduce count printed by fsils_solve stay unchanged.
    static_cast<void>(norm::fsi_ls_norms(mynNo, lhs.commu, residual));
    R = Xg;
    dump_legacy("gmres_solution_final", R);
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

  dump_legacy("bicg_initial_guess", X);
  dump_legacy("bicg_initial_residual", R);
  const int fixed_bicg_iters = forced_schur_bicgstab_iterations(ls);
  for (int i = 0; i < ls.mItr; ++i) {
    const bool hold_for_fixed_iters =
        fixed_bicg_iters > 0 && i < fixed_bicg_iters;
    if (err < eps && !hold_for_fixed_iters) {
      ls.suc = true;
      break;
    }

    apply_schur_operator(P, V);
    if (legacy_dump_enabled && i < legacy_dump_max_iters) {
      dump_legacy((std::string("bicg_iter") + std::to_string(i) + "_P").c_str(), P);
      dump_legacy((std::string("bicg_iter") + std::to_string(i) + "_V").c_str(), V);
    }

    const double denom_alpha = dot::fsils_dot_s(mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) <
        std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) {
      break;
    }
    const double alpha = rho / denom_alpha;

    omp_la::omp_axpby_s(nNo, S, R, -alpha, V);
    apply_schur_operator(S, T);
    if (legacy_dump_enabled && i < legacy_dump_max_iters) {
      dump_legacy((std::string("bicg_iter") + std::to_string(i) + "_S").c_str(), S);
      dump_legacy((std::string("bicg_iter") + std::to_string(i) + "_T").c_str(), T);
    }

    double schur_locals[3] = {
        norm::fsi_ls_norm_sq_local_s(mynNo, S),
        dot::fsils_nc_dot_s(mynNo, T, T),
        dot::fsils_nc_dot_s(mynNo, T, S),
    };
    double schur_globals[3] = {schur_locals[0], schur_locals[1], schur_locals[2]};
    collectives.allreduce_sum(schur_locals, schur_globals, 3);
    const double s_sq = schur_globals[0];
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_s(nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    const double t_sq = schur_globals[1];
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    const double omega = schur_globals[2] / t_sq;

    omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_s(nNo, R, S, -omega, T);

      errO = err;
      double update_locals[2] = {
          norm::fsi_ls_norm_sq_local_s(mynNo, R),
        dot::fsils_nc_dot_s(mynNo, R, Rh),
    };
    double update_globals[2] = {update_locals[0], update_locals[1]};
    collectives.allreduce_sum(update_locals, update_globals, 2);
      err = std::sqrt(std::max(0.0, update_globals[0]));
      const double rhoO = rho;
      rho = update_globals[1];
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
  if (fixed_bicg_iters > 0) {
    ls.suc = (err < eps);
  }

  R = X;
  dump_legacy("bicg_solution_final_pre_postproject", R);
  dump_legacy("bicg_solution_final", R);
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

  static std::uint64_t schur_dump_solve_counter = 0;
  const char* dump_prefix = schur_dump_prefix();
  const std::uint64_t dump_solve_id =
      (dump_prefix == nullptr) ? 0 : schur_dump_solve_counter++;
  const int dump_max_solves = schur_dump_max_solves();
  const int dump_max_iters = schur_dump_max_iters();
  const bool dump_enabled =
      dump_prefix != nullptr &&
      (dump_max_solves < 0 ||
       dump_solve_id < static_cast<std::uint64_t>(dump_max_solves));
  auto dump_schur = [&](const char* label, const Array<double>& values) {
    if (dump_enabled) {
      dump_schur_array_owned(lhs, dump_prefix, dump_solve_id, label, values);
    }
  };

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
    dump_schur("legacy_scalar_solution", R);
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
  const CollectiveOps collectives(lhs.commu);

  auto& cache = schur_cache_entry(ls);
  double pc_setup_time = 0.0;
  const std::uint64_t topology_signature = schur_cache_topology_signature(lhs);
  const std::uint64_t operator_signature = schur_cache_operator_signature(lhs, K, D, G, L);
  if (should_rebuild_schur_cache(cache,
                                 lhs,
                                 ls,
                                 mom_ncomp,
                                 con_ncomp,
                                 topology_signature,
                                 operator_signature)) {
    const double setup_t0 = fe_fsi_linear_solver::fsils_cpu_t();
    cache.preconditioner = build_schur_preconditioner(lhs, ls, mom_ncomp, con_ncomp, K, D, G, L);
    cache.preconditioner.operator_L = &cache.preconditioner.operator_L_storage;
    pc_setup_time = fe_fsi_linear_solver::fsils_cpu_t() - setup_t0;
    cache.valid = true;
    cache.mom_ncomp = mom_ncomp;
    cache.con_ncomp = con_ncomp;
    cache.nNo = nNo;
    cache.nnz = lhs.nnz;
    cache.topology_signature = topology_signature;
    cache.operator_signature = operator_signature;
    cache.schur_preconditioner = ls.schur_preconditioner;
    cache.momentum_approximation = ls.schur_momentum_approximation;
    cache.solves_since_build = 0;
  }
  cache.solves_since_build += 1;
  auto& pc = cache.preconditioner;
  const fe_fsi_linear_solver::HaloExchange schur_halo(lhs);

  auto apply_exact_schur_operator = [&](const Array<double>& in_vec, Array<double>& out_vec) {
    system.GL.apply(
        dso::ghost_synced_input(con_ncomp, in_vec),
        dso::owned_only_output(mom_ncomp, GP),
        dso::owned_only_output(con_ncomp, SP));
    apply_explicit_constraint_schur_corrections(lhs, pc, in_vec, GP, SP);
    schur_halo.sync_owned_to_ghost_vector(mom_ncomp, GP, skip_post_solve_halo_sync());
    HGP = GP;
    gmres::gmres_v(system.momentum, momentum_ls, HGP);
    if (schur_halo.has_owned_halo() && !skip_post_solve_halo_sync()) {
      // The nested momentum solve returns a solution-like overlap vector.
      // Push owner values back to ghosts before reusing it in D*(K^-1*G).
      schur_halo.sync_owned_to_ghost_vector(mom_ncomp, HGP);
    }
    system.D.apply(
        dso::ghost_synced_input(mom_ncomp, HGP),
        dso::owned_only_output(con_ncomp, DGP));
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
  dump_schur("rhs", rhs);
  Array<double> rhs_preconditioned = rhs;
  apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, pc, con_ncomp, nNo, rhs_preconditioned);
  dump_schur("rhs_preconditioned_full", rhs_preconditioned);
  if (dump_enabled) {
    Array<double> rhs_preconditioned_base = rhs;
    apply_base_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, pc, con_ncomp, nNo,
                                    rhs_preconditioned_base);
    sync_schur_halo(lhs, con_ncomp, rhs_preconditioned_base);
    dump_schur("rhs_preconditioned_base", rhs_preconditioned_base);

    Array<double> exact_rhs;
    apply_exact_schur_operator(rhs, exact_rhs);
    dump_schur("exact_operator_rhs", exact_rhs);

    Array<double> exact_rhs_preconditioned = exact_rhs;
    apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, pc, con_ncomp, nNo,
                               exact_rhs_preconditioned);
    dump_schur("exact_operator_rhs_preconditioned", exact_rhs_preconditioned);

    if (con_ncomp == 1) {
      Array<double> global_mean(con_ncomp, nNo);
      global_mean = 0.0;
      for (fsils_int node = 0; node < lhs.mynNo; ++node) {
        global_mean(0, node) = 1.0;
      }
      schur_halo.sync_owned_to_ghost_vector(con_ncomp, global_mean, skip_post_solve_halo_sync());
      dump_schur("probe_global_mean", global_mean);

      Array<double> exact_global_mean;
      apply_exact_schur_operator(global_mean, exact_global_mean);
      dump_schur("exact_operator_global_mean", exact_global_mean);

      Array<double> preconditioned_global_mean = global_mean;
      apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, pc, con_ncomp, nNo,
                                 preconditioned_global_mean);
      dump_schur("preconditioned_global_mean", preconditioned_global_mean);

#if FE_HAS_MPI
      if (lhs.commu.nTasks > 1) {
        std::vector<double> owned_counts(static_cast<std::size_t>(lhs.commu.nTasks), 0.0);
        const double local_owned = static_cast<double>(lhs.mynNo);
        MPI_Allgather(&local_owned,
                      1,
                      cm_mod::mpreal,
                      owned_counts.data(),
                      1,
                      cm_mod::mpreal,
                      lhs.commu.comm);

        std::vector<int> active_ranks;
        active_ranks.reserve(static_cast<std::size_t>(lhs.commu.nTasks));
        for (int rank = 0; rank < lhs.commu.nTasks; ++rank) {
          if (owned_counts[static_cast<std::size_t>(rank)] > 0.5) {
            active_ranks.push_back(rank);
          }
        }
        if (active_ranks.size() >= 2u) {
          const int reference_rank = active_ranks.back();
          const int probe_rank = active_ranks.front();
          const double reference_count =
              owned_counts[static_cast<std::size_t>(reference_rank)];
          if (reference_count > 0.5 && reference_rank != probe_rank) {
            Array<double> partition_probe(con_ncomp, nNo);
            partition_probe = 0.0;
            if (lhs.commu.task == probe_rank) {
              for (fsils_int node = 0; node < lhs.mynNo; ++node) {
                partition_probe(0, node) = 1.0;
              }
            } else if (lhs.commu.task == reference_rank) {
              const double weight =
                  -owned_counts[static_cast<std::size_t>(probe_rank)] / reference_count;
              for (fsils_int node = 0; node < lhs.mynNo; ++node) {
                partition_probe(0, node) = weight;
              }
            }
            schur_halo.sync_owned_to_ghost_vector(con_ncomp, partition_probe,
                                                  skip_post_solve_halo_sync());
            dump_schur("probe_partition_0", partition_probe);

            Array<double> exact_partition_probe;
            apply_exact_schur_operator(partition_probe, exact_partition_probe);
            dump_schur("exact_operator_partition_0", exact_partition_probe);

            Array<double> preconditioned_partition_probe = partition_probe;
            apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, pc, con_ncomp, nNo,
                                       preconditioned_partition_probe);
            dump_schur("preconditioned_partition_0", preconditioned_partition_probe);
          }
        }
      }
#endif
    }
  }
  const bool use_preconditioned_rhs_initial_guess =
      env_enabled("SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS");
  const bool trace_schur_preconditioner_probes =
      env_enabled("SVMP_FSILS_TRACE_SCHUR_PRECONDITIONER_PROBES") && lhs.commu.masF;

  auto trace_schur_preconditioner_probe = [&](const char* label, const Array<double>& q) {
    if (!trace_schur_preconditioner_probes) {
      return;
    }

    const double q_norm = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, q);
    if (!(q_norm > 0.0) || !std::isfinite(q_norm)) {
      return;
    }

    auto compute_solution_mean = [&](const Array<double>& x) -> double {
      if (con_ncomp != 1) {
        return 0.0;
      }
      double local_sum = 0.0;
      double local_count = 0.0;
      for (fsils_int node = 0; node < lhs.mynNo; ++node) {
        local_sum += x(0, node);
        local_count += 1.0;
      }
      double global_sum = local_sum;
      double global_count = local_count;
      collectives.allreduce_sum(local_sum, global_sum);
      collectives.allreduce_sum(local_count, global_count);
      return (global_count > 0.0) ? (global_sum / global_count) : 0.0;
    };

    auto compute_inverse_quality =
        [&](const Array<double>& x, Array<double>& residual_out) -> std::pair<double, double> {
          Array<double> ax;
          apply_exact_schur_operator(x, ax);
          residual_out = ax;
          #pragma omp parallel for schedule(static)
          for (fsils_int node = 0; node < nNo; ++node) {
            for (int comp = 0; comp < con_ncomp; ++comp) {
              residual_out(comp, node) -= q(comp, node);
            }
          }
          const double residual_norm =
              norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, residual_out);
          const double relative =
              residual_norm / std::max(q_norm, std::numeric_limits<double>::min());
          return {residual_norm, relative};
        };

    Array<double> base_solution = q;
    apply_base_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, pc, con_ncomp, nNo, base_solution);
    sync_schur_halo(lhs, con_ncomp, base_solution);
    Array<double> base_residual;
    const auto [base_residual_norm, base_relative] =
        compute_inverse_quality(base_solution, base_residual);

    Array<double> full_solution = q;
    apply_schur_preconditioner(lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, pc, con_ncomp, nNo, full_solution);
    Array<double> full_residual;
    const auto [full_residual_norm, full_relative] =
        compute_inverse_quality(full_solution, full_residual);

    std::fprintf(stderr,
                 "[SCHUR_PC_PROBE] label=%s distributed=%d q_norm=%e base_x_norm=%e base_mean=%e "
                 "base_residual=%e base_rel=%e full_x_norm=%e full_mean=%e full_residual=%e full_rel=%e\n",
                 label,
                 collectives.distributed() ? 1 : 0,
                 q_norm,
                 norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, base_solution),
                 compute_solution_mean(base_solution),
                 base_residual_norm,
                 base_relative,
                 norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, full_solution),
                 compute_solution_mean(full_solution),
                 full_residual_norm,
                 full_relative);
    std::fflush(stderr);
  };

  if (trace_schur_preconditioner_probes) {
    trace_schur_preconditioner_probe("rhs", rhs);

    if (con_ncomp == 1) {
      Array<double> global_mean_probe(con_ncomp, nNo);
      global_mean_probe = 0.0;
      for (fsils_int node = 0; node < lhs.mynNo; ++node) {
        global_mean_probe(0, node) = 1.0;
      }
      trace_schur_preconditioner_probe("constraint_global_mean", global_mean_probe);

#if FE_HAS_MPI
      if (lhs.commu.nTasks > 1) {
        std::vector<double> owned_counts(static_cast<std::size_t>(lhs.commu.nTasks), 0.0);
        const double local_owned = static_cast<double>(lhs.mynNo);
        MPI_Allgather(&local_owned,
                      1,
                      cm_mod::mpreal,
                      owned_counts.data(),
                      1,
                      cm_mod::mpreal,
                      lhs.commu.comm);

        std::vector<int> active_ranks;
        active_ranks.reserve(static_cast<std::size_t>(lhs.commu.nTasks));
        for (int rank = 0; rank < lhs.commu.nTasks; ++rank) {
          if (owned_counts[static_cast<std::size_t>(rank)] > 0.5) {
            active_ranks.push_back(rank);
          }
        }

        if (active_ranks.size() >= 2u) {
          const int reference_rank = active_ranks.back();
          const int probe_rank = active_ranks.front();
          const double reference_count =
              owned_counts[static_cast<std::size_t>(reference_rank)];
          if (reference_count > 0.5 && reference_rank != probe_rank) {
            Array<double> partition_probe(con_ncomp, nNo);
            partition_probe = 0.0;
            if (lhs.commu.task == probe_rank) {
              for (fsils_int node = 0; node < lhs.mynNo; ++node) {
                partition_probe(0, node) = 1.0;
              }
            } else if (lhs.commu.task == reference_rank) {
              const double weight =
                  -owned_counts[static_cast<std::size_t>(probe_rank)] / reference_count;
              for (fsils_int node = 0; node < lhs.mynNo; ++node) {
                partition_probe(0, node) = weight;
              }
            }
            trace_schur_preconditioner_probe("constraint_partition", partition_probe);
          }
        }
      }
#endif
    }

    const int low_rank_probe_count =
        std::min<int>(2, static_cast<int>(pc.low_rank_right.size()));
    for (int probe_idx = 0; probe_idx < low_rank_probe_count; ++probe_idx) {
      trace_schur_preconditioner_probe(
          ("low_rank_right_" + std::to_string(probe_idx)).c_str(),
          pc.low_rank_right[static_cast<std::size_t>(probe_idx)]);
    }
  }

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
    if (use_preconditioned_rhs_initial_guess) {
      Xg = rhs_preconditioned;
    } else {
      Xg = 0.0;
    }
    dump_schur("gmres_initial_guess", Xg);

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

      schur_halo.sync_owned_to_ghost_vector(con_ncomp, Xg, skip_post_solve_halo_sync());
      apply_schur_operator(Xg, ax);
      omp_la::omp_axpby_v(con_ncomp, nNo, residual, rhs_preconditioned, -1.0, ax);
      if (dump_enabled && total_itr == 0) {
        dump_schur("gmres_initial_operator", ax);
        dump_schur("gmres_initial_residual", residual);
      }
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
        schur_halo.sync_owned_to_ghost_vector(con_ncomp, ui, skip_post_solve_halo_sync());
        apply_schur_operator(ui, uip1);
        if (dump_enabled && restart_cycles == 1 && i < dump_max_iters) {
          dump_schur(("gmres_iter" + std::to_string(i) + "_basis").c_str(), ui);
          dump_schur(("gmres_iter" + std::to_string(i) + "_operator").c_str(), uip1);
        }

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
      schur_halo.sync_owned_to_ghost_vector(con_ncomp, Xg, skip_post_solve_halo_sync());
      apply_schur_operator(Xg, ax);
      omp_la::omp_axpby_v(con_ncomp, nNo, residual, rhs_preconditioned, -1.0, ax);
      ls.fNorm = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, residual);
      ls.suc = (ls.fNorm < eps);
    }

    R = Xg;
    dump_schur("gmres_solution_final", R);
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

  if (use_preconditioned_rhs_initial_guess) {
    X = rhs_preconditioned;
    apply_schur_operator(X, R);
    omp_la::omp_axpby_v(con_ncomp, nNo, R, rhs_preconditioned, -1.0, R);
  } else {
    X = 0.0;
    R = rhs_preconditioned;
  }
  dump_schur("bicg_initial_guess", X);
  dump_schur("bicg_initial_residual", R);

  double err = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, R);
  double errO = std::max(err, rhs_norm);
  double rho = err * err;
  double beta = rho;

  P = R;
  Rh = R;
  int i_itr = 1;
  const int fixed_bicg_iters = forced_schur_bicgstab_iterations(ls);

  for (int i = 0; i < ls.mItr; ++i) {
    const bool hold_for_fixed_iters =
        fixed_bicg_iters > 0 && i < fixed_bicg_iters;
    if (err < eps && !hold_for_fixed_iters) {
      ls.suc = true;
      break;
    }

    apply_schur_operator(P, V);
    if (dump_enabled && i < dump_max_iters) {
      dump_schur(("bicg_iter" + std::to_string(i) + "_P").c_str(), P);
      dump_schur(("bicg_iter" + std::to_string(i) + "_V").c_str(), V);
    }

    const double denom_alpha = dot::fsils_dot_v(con_ncomp, mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) <
        std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) {
      break;
    }
    const double alpha = rho / denom_alpha;

    omp_la::omp_axpby_v(con_ncomp, nNo, S, R, -alpha, V);
    apply_schur_operator(S, T);
    if (dump_enabled && i < dump_max_iters) {
      dump_schur(("bicg_iter" + std::to_string(i) + "_S").c_str(), S);
      dump_schur(("bicg_iter" + std::to_string(i) + "_T").c_str(), T);
    }

    double schur_locals[3] = {
        norm::fsi_ls_norm_sq_local_v(con_ncomp, mynNo, S),
        dot::fsils_nc_dot_v(con_ncomp, mynNo, T, T),
        dot::fsils_nc_dot_v(con_ncomp, mynNo, T, S),
    };
    double schur_globals[3] = {schur_locals[0], schur_locals[1], schur_locals[2]};
    collectives.allreduce_sum(schur_locals, schur_globals, 3);
    const double s_sq = schur_globals[0];
    if (std::sqrt(s_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpby_v(con_ncomp, nNo, X, X, alpha, P);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = true;
      ++i_itr;
      break;
    }

    const double t_sq = schur_globals[1];
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) {
      omp_la::omp_axpbypgz_v(con_ncomp, nNo, X, X, alpha, P, 0.0, S);
      R = S;
      errO = err;
      err = std::sqrt(std::max(0.0, s_sq));
      ls.suc = (err < eps);
      ++i_itr;
      break;
    }
    const double omega = schur_globals[2] / t_sq;

    omp_la::omp_axpbypgz_v(con_ncomp, nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_v(con_ncomp, nNo, R, S, -omega, T);

    errO = err;
    double update_locals[2] = {
        norm::fsi_ls_norm_sq_local_v(con_ncomp, mynNo, R),
        dot::fsils_nc_dot_v(con_ncomp, mynNo, R, Rh),
    };
    double update_globals[2] = {update_locals[0], update_locals[1]};
    collectives.allreduce_sum(update_locals, update_globals, 2);
    err = std::sqrt(std::max(0.0, update_globals[0]));

    const double rhoO = rho;
    rho = update_globals[1];

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
  if (fixed_bicg_iters > 0) {
    ls.suc = (err < eps);
  }

  R = X;
  dump_schur("bicg_solution_final", R);
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
  const std::uint64_t topology_signature = schur_cache_topology_signature(lhs);
  const std::uint64_t operator_signature = schur_cache_operator_signature(lhs,
                                                                         *system.momentum_values,
                                                                         *system.D_values,
                                                                         *system.G_values,
                                                                         *system.L_values);
  if (should_rebuild_schur_cache(cache,
                                 lhs,
                                 ls,
                                 system.momentum_components,
                                 system.constraint_components,
                                 topology_signature,
                                 operator_signature)) {
    cache.preconditioner = build_schur_preconditioner(
        lhs, ls, system.momentum_components, system.constraint_components,
        *system.momentum_values, *system.D_values, *system.G_values, *system.L_values);
    cache.preconditioner.operator_L = &cache.preconditioner.operator_L_storage;
    cache.valid = true;
    cache.mom_ncomp = system.momentum_components;
    cache.con_ncomp = system.constraint_components;
    cache.nNo = nNo;
    cache.nnz = lhs.nnz;
    cache.topology_signature = topology_signature;
    cache.operator_signature = operator_signature;
    cache.schur_preconditioner = ls.schur_preconditioner;
    cache.momentum_approximation = ls.schur_momentum_approximation;
    cache.solves_since_build = 0;
  }
  cache.solves_since_build += 1;

  apply_schur_preconditioner(
      lhs.rowPtr, lhs.colPtr, lhs.diagPtr, lhs, cache.preconditioner,
      system.constraint_components, nNo, R);
}

} // namespace bicgs
