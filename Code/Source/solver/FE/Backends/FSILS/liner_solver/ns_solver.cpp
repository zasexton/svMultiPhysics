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

// Fractional-step block Schur complement solver for saddle-point systems.
// Form: AU=R, where A = [K G; D L] with exact analytical D (not -G^T).
// Uses asymmetric BiCGStab Schur complement instead of symmetric CG.
// Supports arbitrary momentum (field-A) and constraint (field-B) block positions
// via block layout metadata in the FSILS_lsType structure.

#include "ns_solver.h"

#include "fsils_api.hpp"
#include "fils_struct.hpp"

#include "add_bc_mul.h"
#include "bicgs.h"
#include "block_schur_strategy_selector.h"
#include "distributed_mpi_ops.h"
#include "distributed_low_rank_correction.h"
#include "distributed_sparse_operator.h"
#include "dot.h"
#include "ge.h"
#include "gmres.h"
#include "norm.h"
#include "omp_la.h"

#include "Array3.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>

namespace ns_solver {

namespace dso = fe_fsi_linear_solver::distributed_sparse_operator;

using fe_fsi_linear_solver::fsils_int;
using DistributedLowRankCorrection =
    fe_fsi_linear_solver::distributed_low_rank_correction::DistributedLowRankCorrection;

namespace {

constexpr int kNativeFaceDuplicateCouplingId = -2;

[[nodiscard]] bool envFlagEnabled(const char* env_name) noexcept
{
  const char* env = std::getenv(env_name);
  if (env == nullptr) {
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

[[nodiscard]] int envIntValue(const char* env_name, int default_value) noexcept
{
  const char* env = std::getenv(env_name);
  if (env == nullptr || *env == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  return (end == env) ? default_value : static_cast<int>(value);
}

[[nodiscard]] int blockSchurMinOuterIterations() noexcept
{
  return std::max(0, envIntValue("SVMP_FSILS_BLOCKSCHUR_MIN_OUTER_ITERS", 0));
}

[[nodiscard]] bool fsilsTraceEnabled() noexcept
{
  return envFlagEnabled("SVMP_FSILS_TRACE") ||
         envFlagEnabled("SVMP_FSILS_NS_TRACE") ||
         envFlagEnabled("SVMP_FSILS_NS_SOLVER_TRACE");
}

[[nodiscard]] bool fsilsFullColumnCompareEnabled() noexcept
{
  return envFlagEnabled("SVMP_FSILS_NS_FULL_COLUMN_COMPARE");
}

[[nodiscard]] const char* fsilsOracleBlockCompareFile() noexcept
{
  const char* path = std::getenv("SVMP_FSILS_NS_ORACLE_BLOCK_COMPARE_FILE");
  if (path == nullptr || *path == '\0') {
    return nullptr;
  }
  return path;
}

[[nodiscard]] bool skipPostSolveHaloSync() noexcept
{
  return envFlagEnabled("SVMP_FSILS_SKIP_POST_SOLVE_COMM");
}

[[nodiscard]] const char* fsilsNsDumpPrefix() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_NS_DUMP_PREFIX");
  if (env == nullptr) {
    return nullptr;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  return (*env == '\0') ? nullptr : env;
}

[[nodiscard]] int fsilsNsDumpMaxSolves() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_NS_DUMP_MAX_SOLVES");
  if (env == nullptr || *env == '\0') {
    return 1;
  }
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  return (end == env) ? 1 : static_cast<int>(value);
}

void dumpScalarVectorByGlobalNode(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                  const char* prefix,
                                  std::uint64_t solve_id,
                                  int outer_iter,
                                  const char* label,
                                  const Vector<double>& values)
{
  if (prefix == nullptr || *prefix == '\0') {
    return;
  }

  std::ostringstream path;
  path << prefix << ".solve" << solve_id
       << ".iter" << outer_iter
       << "." << label
       << ".rank" << lhs.commu.task << ".txt";
  std::ofstream out(path.str());
  if (!out) {
    return;
  }

  out << std::setprecision(17) << std::scientific;
  out << "# task " << lhs.commu.task
      << " nTasks " << lhs.commu.nTasks
      << " solve " << solve_id
      << " iter " << outer_iter
      << " label " << label
      << " mynNo " << lhs.mynNo
      << " nNo " << lhs.nNo << '\n';
  const bool has_debug_global_nodes = lhs.debug_global_nodes.size() == lhs.nNo;
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

bool loadOracleScalarMode(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                          const char* path,
                          Vector<double>& values)
{
  if (path == nullptr || *path == '\0') {
    return false;
  }

  std::ifstream in(path);
  if (!in) {
    return false;
  }

  std::map<int, double> by_global_node;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    int global_node = -1;
    double value = 0.0;
    if (!(iss >> global_node >> value)) {
      continue;
    }
    by_global_node[global_node] = value;
  }
  if (by_global_node.empty()) {
    return false;
  }

  values = 0.0;
  bool found_any = false;
  for (fsils_int old = 0; old < lhs.nNo; ++old) {
    if (old < 0 || old >= lhs.gNodes.size()) {
      continue;
    }
    const auto it = by_global_node.find(lhs.gNodes(old));
    if (it == by_global_node.end()) {
      continue;
    }
    const int internal = lhs.map(old);
    if (internal < 0 || internal >= values.size()) {
      continue;
    }
    values(internal) = it->second;
    found_any = true;
  }
  return found_any;
}

[[nodiscard]] bool syncOrderTraceEnabled() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_TRACE_SYNC_ORDER");
  if (env == nullptr) {
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

void traceSyncOrder(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                    const char* label,
                    int outer_iter,
                    fsils_int mynNo,
                    fsils_int nNo,
                    double fNorm) noexcept
{
  if (!syncOrderTraceEnabled()) {
    return;
  }
  std::fprintf(stderr,
               "[NS_SYNC_TRACE] task=%d label=%s outer=%d mynNo=%lld nNo=%lld owned_halo_neighbors=%zu shnNo=%lld fNorm=%.17e skip=%d\n",
               lhs.commu.task,
               label,
               outer_iter,
               static_cast<long long>(mynNo),
               static_cast<long long>(nNo),
               lhs.owned_halo_neighbor_ranks.size(),
               static_cast<long long>(lhs.shnNo),
               fNorm,
               skipPostSolveHaloSync() ? 1 : 0);
  std::fflush(stderr);
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

double scalar_vector_mean(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                          const Vector<double>& values)
{
  long double local_sum = 0.0L;
  long double local_count = 0.0L;
  for (fsils_int node = 0; node < lhs.mynNo; ++node) {
    local_sum += static_cast<long double>(values(node));
    local_count += 1.0L;
  }

  long double global_sum = local_sum;
  long double global_count = local_count;
#if FE_HAS_MPI
  if (lhs.commu.nTasks > 1) {
    fe_fsi_linear_solver::fsils_allreduce_sum(
        &local_sum, &global_sum, 1, MPI_LONG_DOUBLE,
        const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
    fe_fsi_linear_solver::fsils_allreduce_sum(
        &local_count, &global_count, 1, MPI_LONG_DOUBLE,
        const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
  }
#endif

  return (global_count > 0.0L) ? static_cast<double>(global_sum / global_count) : 0.0;
}

struct ScalarZeroMeanStats {
  double mean_before{0.0};
  double mean_after{0.0};
  double max_abs_shift{0.0};
};

ScalarZeroMeanStats subtract_scalar_global_mean(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    Vector<double>& values)
{
  ScalarZeroMeanStats stats{};
  if (lhs.mynNo <= 0) {
    return stats;
  }

  stats.mean_before = scalar_vector_mean(lhs, values);
  const double shift = stats.mean_before;
  #pragma omp parallel for schedule(static)
  for (fsils_int node = 0; node < lhs.mynNo; ++node) {
    values(node) -= shift;
  }
  stats.mean_after = scalar_vector_mean(lhs, values);
  stats.max_abs_shift = std::abs(shift);
  return stats;
}

struct ScalarPartitionMeanStats {
  double global_mean_before{0.0};
  double local_mean_min_before{0.0};
  double local_mean_max_before{0.0};
  double global_mean_after{0.0};
  double local_mean_min_after{0.0};
  double local_mean_max_after{0.0};
  double max_abs_shift{0.0};
};

ScalarPartitionMeanStats equalize_scalar_partition_means(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    Vector<double>& values)
{
  ScalarPartitionMeanStats stats{};
  if (lhs.commu.nTasks <= 1 || lhs.mynNo <= 0) {
    return stats;
  }

  long double local_sum = 0.0L;
  long double local_count = static_cast<long double>(lhs.mynNo);
  for (fsils_int node = 0; node < lhs.mynNo; ++node) {
    local_sum += static_cast<long double>(values(node));
  }

  long double global_sum = local_sum;
  long double global_count = local_count;
#if FE_HAS_MPI
  fe_fsi_linear_solver::fsils_allreduce_sum(
      &local_sum, &global_sum, 1, MPI_LONG_DOUBLE,
      const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
  fe_fsi_linear_solver::fsils_allreduce_sum(
      &local_count, &global_count, 1, MPI_LONG_DOUBLE,
      const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
#endif

  const double local_mean_before =
      (local_count > 0.0L) ? static_cast<double>(local_sum / local_count) : 0.0;
  const double global_mean =
      (global_count > 0.0L) ? static_cast<double>(global_sum / global_count) : 0.0;

  stats.global_mean_before = global_mean;
  stats.local_mean_min_before = local_mean_before;
  stats.local_mean_max_before = local_mean_before;
#if FE_HAS_MPI
  fe_fsi_linear_solver::fsils_allreduce_in_place(
      &stats.local_mean_min_before, 1, cm_mod::mpreal, MPI_MIN,
      const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
  fe_fsi_linear_solver::fsils_allreduce_in_place(
      &stats.local_mean_max_before, 1, cm_mod::mpreal, MPI_MAX,
      const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
#endif

  const double shift = local_mean_before - global_mean;
  stats.max_abs_shift = std::abs(shift);
#if FE_HAS_MPI
  fe_fsi_linear_solver::fsils_allreduce_in_place(
      &stats.max_abs_shift, 1, cm_mod::mpreal, MPI_MAX,
      const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
#endif

  for (fsils_int node = 0; node < lhs.mynNo; ++node) {
    values(node) -= shift;
  }

  long double local_sum_after = 0.0L;
  for (fsils_int node = 0; node < lhs.mynNo; ++node) {
    local_sum_after += static_cast<long double>(values(node));
  }
  const double local_mean_after =
      (local_count > 0.0L) ? static_cast<double>(local_sum_after / local_count) : 0.0;
  stats.global_mean_after = scalar_vector_mean(lhs, values);
  stats.local_mean_min_after = local_mean_after;
  stats.local_mean_max_after = local_mean_after;
#if FE_HAS_MPI
  fe_fsi_linear_solver::fsils_allreduce_in_place(
      &stats.local_mean_min_after, 1, cm_mod::mpreal, MPI_MIN,
      const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
  fe_fsi_linear_solver::fsils_allreduce_in_place(
      &stats.local_mean_max_after, 1, cm_mod::mpreal, MPI_MAX,
      const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu));
#endif
  return stats;
}

double galerkin_residual_norm_sq(const Array<double>& A,
                                 const Vector<double>& B,
                                 const Vector<double>& x,
                                 int active_cols,
                                 double initial_norm_sq)
{
  long double quad = 0.0L;
  long double lin = 0.0L;
  for (int i = 0; i < active_cols; ++i) {
    const long double xi = static_cast<long double>(x(i));
    lin += xi * static_cast<long double>(B(i));
    for (int j = 0; j < active_cols; ++j) {
      quad += xi * static_cast<long double>(A(j, i)) *
              static_cast<long double>(x(j));
    }
  }
  return std::max(
      0.0,
      static_cast<double>(static_cast<long double>(initial_norm_sq) - 2.0L * lin + quad));
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

[[nodiscard]] double combined_block_norm_sq(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                            int mom_ncomp,
                                            const Array<double>& momentum,
                                            bool scalar_constraint,
                                            int con_ncomp,
                                            const Array<double>* constraint_array,
                                            const Vector<double>* constraint_vector) noexcept
{
  double local_values[2] = {0.0, 0.0};
  if (mom_ncomp > 0) {
    local_values[0] = norm::fsi_ls_norm_sq_local_v(mom_ncomp, lhs.mynNo, momentum);
  }
  if (scalar_constraint) {
    if (constraint_vector != nullptr) {
      local_values[1] = norm::fsi_ls_norm_sq_local_s(lhs.mynNo, *constraint_vector);
    }
  } else if (constraint_array != nullptr) {
    local_values[1] = norm::fsi_ls_norm_sq_local_v(con_ncomp, lhs.mynNo, *constraint_array);
  }
  double global_values[2] = {local_values[0], local_values[1]};
  const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
  collectives.allreduce_sum(local_values, global_values, 2);
  return global_values[0] + global_values[1];
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

bool invert_dense_matrix(int n,
                         const std::vector<double>& matrix,
                         std::vector<double>& inverse)
{
  if (n <= 0 || matrix.size() != static_cast<std::size_t>(n * n)) {
    inverse.clear();
    return false;
  }

  std::vector<double> a = matrix;
  inverse.assign(static_cast<std::size_t>(n * n), 0.0);
  for (int i = 0; i < n; ++i) {
    inverse[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(i)] = 1.0;
  }

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double pivot_abs = std::abs(a[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
                                  static_cast<std::size_t>(col)]);
    for (int row = col + 1; row < n; ++row) {
      const double cand =
          std::abs(a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                     static_cast<std::size_t>(col)]);
      if (cand > pivot_abs) {
        pivot = row;
        pivot_abs = cand;
      }
    }

    if (!(pivot_abs > 1e-30)) {
      inverse.clear();
      return false;
    }

    if (pivot != col) {
      for (int j = 0; j < n; ++j) {
        std::swap(a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n) +
                    static_cast<std::size_t>(j)],
                  a[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
                    static_cast<std::size_t>(j)]);
        std::swap(inverse[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n) +
                          static_cast<std::size_t>(j)],
                  inverse[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
                          static_cast<std::size_t>(j)]);
      }
    }

    const double diag =
        a[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
          static_cast<std::size_t>(col)];
    const double inv_diag = 1.0 / diag;
    for (int j = 0; j < n; ++j) {
      a[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
        static_cast<std::size_t>(j)] *= inv_diag;
      inverse[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(j)] *= inv_diag;
    }

    for (int row = 0; row < n; ++row) {
      if (row == col) {
        continue;
      }
      const double factor =
          a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(col)];
      if (std::abs(factor) <= 1e-30) {
        continue;
      }
      for (int j = 0; j < n; ++j) {
        a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
          static_cast<std::size_t>(j)] -=
            factor *
            a[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(j)];
        inverse[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                static_cast<std::size_t>(j)] -=
            factor *
            inverse[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) +
                    static_cast<std::size_t>(j)];
      }
    }
  }

  return true;
}

void build_explicit_block_corrections(
    const fe_fsi_linear_solver::FSILS_lhsType& lhs,
    int mom_start,
    int mom_ncomp,
    int con_start,
    int con_ncomp,
    DistributedLowRankCorrection& correction)
{
  correction =
      fe_fsi_linear_solver::distributed_low_rank_correction::build(
          lhs, mom_start, mom_ncomp, con_start, con_ncomp);
}

void apply_constraint_block_corrections(
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const DistributedLowRankCorrection& correction,
    const Array<double>& in_constraint,
    Array<double>& out_momentum,
    Array<double>& out_constraint)
{
  fe_fsi_linear_solver::distributed_low_rank_correction::apply_constraint_driven(
      lhs, correction, in_constraint, out_momentum, out_constraint);
}

void apply_momentum_block_corrections(
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const DistributedLowRankCorrection& correction,
    const Array<double>& in_momentum,
    Array<double>& out_constraint)
{
  fe_fsi_linear_solver::distributed_low_rank_correction::apply_momentum_driven(
      lhs, correction, in_momentum, out_constraint);
}

[[nodiscard]] bool has_coupled_block_corrections(const fe_fsi_linear_solver::FSILS_lhsType& lhs)
{
  return std::any_of(lhs.face.begin(), lhs.face.end(),
      [](const fe_fsi_linear_solver::FSILS_faceType& face) {
        return face.coupledFlag && std::abs(face.res) > 1e-30;
      });
}

void apply_full_coupled_operator(
    const fe_fsi_linear_solver::distributed_solver_bundles::VectorLinearSystem& system,
    const Array<double>& x,
    Array<double>& y)
{
  auto& lhs = *system.lhs;
  system.A.apply(
      dso::ghost_synced_input(system.components, x),
      dso::owned_only_output(system.components, y));
  add_bc_mul::add_bc_mul(lhs, add_bc_mul::BcopType::BCOP_TYPE_ADD, system.components, x, y);
}

void apply_full_coupled_operator(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                 int dof,
                                 const Array<double>& Val,
                                 const Array<double>& x,
                                 Array<double>& y)
{
  apply_full_coupled_operator(
      fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, dof, Val),
      x, y);
}

void apply_full_coupled_outer_operator(
    const fe_fsi_linear_solver::distributed_solver_bundles::VectorLinearSystem& system,
    const Array<double>& x,
    Array<double>& y)
{
  apply_full_coupled_operator(system, x, y);
}

void apply_full_coupled_outer_operator(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                       int dof,
                                       const Array<double>& Val,
                                       const Array<double>& x,
                                       Array<double>& y)
{
  apply_full_coupled_outer_operator(
      fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, dof, Val),
      x, y);
}

void apply_full_coupled_outer_residual_preconditioner(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                                      int dof,
                                                      Array<double>& y)
{
  (void)lhs;
  (void)dof;
  (void)y;
}

void split_scalar_block_vector(const Array<double>& in,
                               int mom_start,
                               int mom_ncomp,
                               int con_start,
                               Array<double>& momentum,
                               Vector<double>& constraint)
{
  const fsils_int nNo = in.ncols();
  momentum.resize(mom_ncomp, nNo);
  constraint.resize(nNo);
  #pragma omp parallel for schedule(static)
  for (fsils_int node = 0; node < nNo; ++node) {
    for (int comp = 0; comp < mom_ncomp; ++comp) {
      momentum(comp, node) = in(mom_start + comp, node);
    }
    constraint(node) = in(con_start, node);
  }
}

struct ScalarTraceStats {
  double l2{0.0};
  double mean{0.0};
  double min{0.0};
  double max{0.0};
};

struct VectorTraceStats {
  double l2{0.0};
  double max_abs{0.0};
};

[[nodiscard]] ScalarTraceStats compute_scalar_trace_stats(
    const fsils_int mynNo,
    fe_fsi_linear_solver::FSILS_commuType& commu,
    const Vector<double>& vec)
{
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
  fe_fsi_linear_solver::fsils_allreduce_sum_in_place(&global_sq_sum, 1, cm_mod::mpreal, commu);
  fe_fsi_linear_solver::fsils_allreduce_sum_in_place(&global_sum, 1, cm_mod::mpreal, commu);
  fe_fsi_linear_solver::fsils_allreduce_sum_in_place(&global_count, 1, cm_mod::mpreal, commu);

  double global_min = local_min;
  double global_max = local_max;
  if (global_count > 0.0) {
    fe_fsi_linear_solver::fsils_allreduce_in_place(&global_min, 1, cm_mod::mpreal, MPI_MIN, commu);
    fe_fsi_linear_solver::fsils_allreduce_in_place(&global_max, 1, cm_mod::mpreal, MPI_MAX, commu);
  } else {
    global_min = 0.0;
    global_max = 0.0;
  }

  ScalarTraceStats stats;
  stats.l2 = std::sqrt(std::max(0.0, global_sq_sum));
  stats.mean = (global_count > 0.0) ? (global_sum / global_count) : 0.0;
  stats.min = global_min;
  stats.max = global_max;
  return stats;
}

[[nodiscard]] VectorTraceStats compute_vector_trace_stats(
    const int ncomp,
    const fsils_int mynNo,
    fe_fsi_linear_solver::FSILS_commuType& commu,
    const Array<double>& vec)
{
  VectorTraceStats stats;
  stats.l2 = norm::fsi_ls_normv(ncomp, mynNo, commu, vec);

  double local_max_abs = 0.0;
  for (fsils_int node = 0; node < mynNo; ++node) {
    for (int comp = 0; comp < ncomp; ++comp) {
      local_max_abs = std::max(local_max_abs, std::abs(vec(comp, node)));
    }
  }
  stats.max_abs = local_max_abs;
  fe_fsi_linear_solver::fsils_allreduce_in_place(&stats.max_abs, 1, cm_mod::mpreal, MPI_MAX, commu);
  return stats;
}

void trace_scalar_block_vector_stats(bool master_flag,
                                     const char* label,
                                     const ScalarTraceStats& stats,
                                     double dot_with_rhs = std::numeric_limits<double>::quiet_NaN())
{
  if (!master_flag) {
    return;
  }
  if (std::isnan(dot_with_rhs)) {
    fprintf(stderr,
            "[NS_SOLVER] %s l2=%e mean=%e min=%e max=%e\n",
            label,
            stats.l2,
            stats.mean,
            stats.min,
            stats.max);
    return;
  }

  fprintf(stderr,
          "[NS_SOLVER] %s l2=%e mean=%e min=%e max=%e dot_rhs=%e\n",
          label,
          stats.l2,
          stats.mean,
          stats.min,
          stats.max,
          dot_with_rhs);
}

void trace_momentum_block_vector_stats(bool master_flag,
                                       const char* label,
                                       const VectorTraceStats& stats)
{
  if (!master_flag) {
    return;
  }
  fprintf(stderr,
          "[NS_SOLVER] %s l2=%e max_abs=%e\n",
          label,
          stats.l2,
          stats.max_abs);
}

void trace_compare_scalar_block_column(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                       int dof,
                                       const Array<double>& Val,
                                       int mom_start,
                                       int mom_ncomp,
                                       int con_start,
                                       const Array<double>& trial_momentum,
                                       const Vector<double>& trial_constraint,
                                       const Array<double>& expected_momentum,
  const Vector<double>& expected_constraint,
  const char* label)
{
  if (!fsilsTraceEnabled()) {
    return;
  }

  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;

  Array<double> trial(dof, nNo);
  trial = 0.0;
  #pragma omp parallel for schedule(static)
  for (fsils_int node = 0; node < nNo; ++node) {
    for (int comp = 0; comp < mom_ncomp; ++comp) {
      trial(mom_start + comp, node) = trial_momentum(comp, node);
    }
    trial(con_start, node) = trial_constraint(node);
  }

  Array<double> full_applied(dof, nNo);
  apply_full_coupled_operator(lhs, dof, Val, trial, full_applied);

  Array<double> full_momentum;
  Vector<double> full_constraint;
  split_scalar_block_vector(full_applied, mom_start, mom_ncomp, con_start,
                            full_momentum, full_constraint);

  Array<double> diff_momentum = full_momentum;
  Vector<double> diff_constraint = full_constraint;
  #pragma omp parallel for schedule(static)
  for (fsils_int node = 0; node < nNo; ++node) {
    for (int comp = 0; comp < mom_ncomp; ++comp) {
      diff_momentum(comp, node) -= expected_momentum(comp, node);
    }
    diff_constraint(node) -= expected_constraint(node);
  }

  const double expected_norm =
      std::hypot(norm::fsi_ls_normv(mom_ncomp, mynNo, lhs.commu, expected_momentum),
                 norm::fsi_ls_norms(mynNo, lhs.commu, expected_constraint));
  const double full_norm =
      std::hypot(norm::fsi_ls_normv(mom_ncomp, mynNo, lhs.commu, full_momentum),
                 norm::fsi_ls_norms(mynNo, lhs.commu, full_constraint));
  const double diff_norm =
      std::hypot(norm::fsi_ls_normv(mom_ncomp, mynNo, lhs.commu, diff_momentum),
                 norm::fsi_ls_norms(mynNo, lhs.commu, diff_constraint));
  const double rel = (expected_norm > 0.0) ? diff_norm / expected_norm : diff_norm;

  if (lhs.commu.masF) {
    fprintf(stderr,
            "[NS_SOLVER] full-column compare %s |expected|=%e |full|=%e |diff|=%e rel=%e\n",
            label,
            expected_norm,
            full_norm,
            diff_norm,
            rel);
  }
}

void assemble_scalar_block_vector(const Array<double>& momentum,
                                  const Vector<double>& constraint,
                                  int dof,
                                  int mom_start,
                                  int mom_ncomp,
                                  int con_start,
                                  Array<double>& out)
{
  const fsils_int nNo = momentum.ncols();
  if (out.nrows() != dof || out.ncols() != nNo) {
    out.resize(dof, nNo);
  }
  out = 0.0;
  #pragma omp parallel for schedule(static)
  for (fsils_int node = 0; node < nNo; ++node) {
    for (int comp = 0; comp < mom_ncomp; ++comp) {
      out(mom_start + comp, node) = momentum(comp, node);
    }
    out(con_start, node) = constraint(node);
  }
}

bool ns_solver_coupled_fgmres_scalar(
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    fe_fsi_linear_solver::FSILS_lsType& ls,
    int dof,
    int mom_start,
    int mom_ncomp,
    int con_start,
    const Array<double>& Val,
    const Array<double>& mK,
    const Array<double>& mD,
    const Array<double>& mG,
    const Vector<double>& mL,
    const DistributedLowRankCorrection& explicit_low_rank_correction,
    Array<double>& Ri)
{
  using namespace fe_fsi_linear_solver;

  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;
  const HaloExchange halo(lhs);
  const auto full_system =
      fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, dof, Val);
  const auto momentum_system =
      fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, mom_ncomp, mK);
  const auto schur_system =
      fe_fsi_linear_solver::distributed_solver_bundles::make_scalar_block_schur_system(
          lhs, mom_ncomp, mK, mD, mG, mL);
  const int pressure_ncomp = 1;
  const int default_outer_krylov_dim =
      std::max(ls.RI.mItr, ls.RI.mItr * std::max(1, ls.GM.mItr));
  const int outer_krylov_dim =
      std::max(1, (ls.RI.sD > 0) ? ls.RI.sD : default_outer_krylov_dim);

  ls.RI.ws.ensure_gmres_v(dof, nNo, outer_krylov_dim);
  auto& h = ls.RI.ws.h;
  auto& v_basis = ls.RI.ws.u3;
  auto& x = ls.RI.ws.X2;
  auto& y_coeff = ls.RI.ws.y;
  auto& c = ls.RI.ws.c;
  auto& s = ls.RI.ws.s;
  auto& err = ls.RI.ws.err;

  Array3<double> z_basis(dof, nNo, outer_krylov_dim);
  Array<double> residual(dof, nNo);
  Array<double> ax(dof, nNo);
  Array<double> w(dof, nNo);

  Array<double> mom_rhs(mom_ncomp, nNo);
  Array<double> mom_sol(mom_ncomp, nNo);
  Array<double> mom_tmp(mom_ncomp, nNo);
  Array<double> gp(mom_ncomp, nNo);
  Array<double> dummy_constraint(pressure_ncomp, nNo);
  Vector<double> con_rhs(nNo);
  Vector<double> con_tmp(nNo);
  Vector<double> con_sol(nNo);

  const auto rhs = Ri;
  const auto collective_before_total = lhs.commu.collective_stats;
  const double callD_before = ls.RI.callD;

  auto apply_blockschur_preconditioner = [&](const Array<double>& in_vec, Array<double>& out_vec) {
    split_scalar_block_vector(in_vec, mom_start, mom_ncomp, con_start, mom_rhs, con_rhs);

    mom_sol = mom_rhs;
    gmres::gmres_v(momentum_system, ls.GM, mom_sol);
    halo.sync_owned_to_ghost_vector(mom_ncomp, mom_sol, skipPostSolveHaloSync());

    schur_system.D.apply(
        dso::ghost_synced_input(mom_ncomp, mom_sol),
        dso::owned_only_output(con_tmp));
    copy_scalar_vector_to_array(con_tmp, dummy_constraint);
    apply_momentum_block_corrections(lhs, explicit_low_rank_correction, mom_sol, dummy_constraint);
    copy_scalar_array_to_vector(dummy_constraint, con_tmp);

    #pragma omp parallel for schedule(static)
    for (fsils_int node = 0; node < nNo; ++node) {
      con_sol(node) = con_rhs(node) - con_tmp(node);
    }

    bicgs::schur_precondition(schur_system, ls.CG, con_sol);

    schur_system.G.apply(
        dso::ghost_synced_input(con_sol),
        dso::owned_only_output(mom_ncomp, gp));
    copy_scalar_vector_to_array(con_sol, dummy_constraint);
    Array<double> dummy_lp(pressure_ncomp, nNo);
    dummy_lp = 0.0;
    apply_constraint_block_corrections(lhs, explicit_low_rank_correction, dummy_constraint, gp,
                                       dummy_lp);

    #pragma omp parallel for schedule(static)
    for (fsils_int node = 0; node < nNo; ++node) {
      for (int comp = 0; comp < mom_ncomp; ++comp) {
        mom_tmp(comp, node) = mom_rhs(comp, node) - gp(comp, node);
      }
    }

    mom_sol = mom_tmp;
    gmres::gmres_v(momentum_system, ls.GM, mom_sol);
    halo.sync_owned_to_ghost_vector(mom_ncomp, mom_sol, skipPostSolveHaloSync());

    assemble_scalar_block_vector(mom_sol, con_sol, dof, mom_start, mom_ncomp, con_start, out_vec);
  };

  auto compute_true_residual = [&](const Array<double>& trial, Array<double>& out_residual) {
    apply_full_coupled_operator(full_system, trial, ax);
    #pragma omp parallel for schedule(static)
    for (fsils_int node = 0; node < nNo; ++node) {
      for (int comp = 0; comp < dof; ++comp) {
        out_residual(comp, node) = rhs(comp, node) - ax(comp, node);
      }
    }
  };

  x = 0.0;
  residual = rhs;
  const double true_initial_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, residual);
  apply_full_coupled_outer_residual_preconditioner(lhs, dof, residual);
  const double outer_initial_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, residual);

  ls.RI.iNorm = true_initial_norm;
  const double true_initial_norm_sq = true_initial_norm * true_initial_norm;
  ls.RI.fNorm = true_initial_norm_sq;
  ls.RI.dB = true_initial_norm_sq;

  const double true_eps = std::max(ls.RI.absTol, ls.RI.relTol * true_initial_norm);
  const double outer_eps = std::max(ls.RI.absTol, ls.RI.relTol * outer_initial_norm);
  if (true_initial_norm <= true_eps) {
    Ri = x;
    ls.RI.itr = 0;
    ls.RI.suc = true;
    ls.RI.fNorm = true_initial_norm;
    ls.RI.callD = fsils_cpu_t() - ls.RI.callD;
    ls.RI.stats.record_call(0,
                            /*restart_cycles=*/1,
                            fsils_collective_delta(collective_before_total, lhs.commu.collective_stats),
                            /*setup_seconds=*/0.0,
                            ls.RI.callD - callD_before);
    return true;
  }
  if (!(outer_initial_norm > std::numeric_limits<double>::epsilon())) {
    ls.RI.callD = fsils_cpu_t() - ls.RI.callD;
    ls.RI.suc = false;
    return false;
  }

  h = 0.0;
  for (int i = 0; i <= outer_krylov_dim; ++i) {
    err(i) = 0.0;
    if (i < outer_krylov_dim) {
      c(i) = 0.0;
      s(i) = 0.0;
      y_coeff(i) = 0.0;
    }
  }
  err(0) = outer_initial_norm;

  auto v0 = v_basis.rslice(0);
  v0 = residual;
  omp_la::omp_mul_v(dof, nNo, 1.0 / outer_initial_norm, v0);

  int last_i = -1;
  for (int i = 0; i < outer_krylov_dim; ++i) {
    const auto collective_before_outer = lhs.commu.collective_stats;
    last_i = i;

    auto vi = v_basis.rslice(i);
    auto zi = z_basis.rslice(i);
    auto vip1 = v_basis.rslice(i + 1);

    apply_blockschur_preconditioner(vi, zi);
    apply_full_coupled_outer_operator(full_system, zi, w);
    vip1 = w;

    for (int j = 0; j <= i; ++j) {
      h(j, i) = dot::fsils_dot_v(dof, mynNo, lhs.commu, v_basis.rslice(j), vip1);
      omp_la::omp_sum_v(dof, nNo, -h(j, i), vip1, v_basis.rslice(j));
    }

    h(i + 1, i) = norm::fsi_ls_normv(dof, mynNo, lhs.commu, vip1);
    const bool breakdown =
        !(h(i + 1, i) > std::numeric_limits<double>::epsilon() * std::max(ls.RI.iNorm, 1.0));
    if (!breakdown) {
      omp_la::omp_mul_v(dof, nNo, 1.0 / h(i + 1, i), vip1);
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
    err(i + 1) = -s(i) * err(i);
    err(i) = c(i) * err(i);
    ls.RI.itr = i + 1;
    ls.blockschur_stats.record_outer_iteration(
        fsils_collective_delta(collective_before_outer, lhs.commu.collective_stats));

    if (std::abs(err(i + 1)) < outer_eps || breakdown) {
      break;
    }
  }

  if (last_i < 0) {
    return false;
  }

  for (int i = 0; i <= last_i; ++i) {
    y_coeff(i) = err(i);
  }
  for (int j = last_i; j >= 0; --j) {
    for (int k = j + 1; k <= last_i; ++k) {
      y_coeff(j) -= h(j, k) * y_coeff(k);
    }
    if (std::abs(h(j, j)) > std::numeric_limits<double>::epsilon()) {
      y_coeff(j) /= h(j, j);
    } else {
      y_coeff(j) = 0.0;
    }
  }

  x = 0.0;
  for (int j = 0; j <= last_i; ++j) {
    if (std::abs(y_coeff(j)) > 1e-30) {
      omp_la::omp_sum_v(dof, nNo, y_coeff(j), x, z_basis.rslice(j));
    }
  }

  compute_true_residual(x, residual);
  split_scalar_block_vector(residual, mom_start, mom_ncomp, con_start, mom_rhs, con_rhs);
  double final_norm_locals[2] = {
      norm::fsi_ls_norm_sq_local_v(mom_ncomp, lhs.mynNo, mom_rhs),
      norm::fsi_ls_norm_sq_local_s(lhs.mynNo, con_rhs),
  };
  double final_norm_globals[2] = {final_norm_locals[0], final_norm_locals[1]};
  {
    const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
    collectives.allreduce_sum(final_norm_locals, final_norm_globals, 2);
  }
  const double momentum_norm = std::sqrt(std::max(0.0, final_norm_globals[0]));
  const double constraint_norm = std::sqrt(std::max(0.0, final_norm_globals[1]));
  const double final_norm_sq = final_norm_globals[0] + final_norm_globals[1];

  ls.RI.fNorm = std::sqrt(final_norm_sq);
  ls.RI.suc = (ls.RI.fNorm < true_eps);
  ls.Resc = (final_norm_sq > 0.0) ? static_cast<int>(100.0 * (constraint_norm * constraint_norm) / final_norm_sq) : 0;
  ls.Resm = 100 - ls.Resc;
  ls.RI.dB = (true_initial_norm > std::numeric_limits<double>::epsilon() && ls.RI.fNorm > 0.0)
      ? 10.0 * std::log(ls.RI.fNorm / true_initial_norm)
      : 0.0;
  ls.RI.callD = fsils_cpu_t() - ls.RI.callD;
  ls.RI.stats.record_call(ls.RI.itr,
                          /*restart_cycles=*/1,
                          fsils_collective_delta(collective_before_total, lhs.commu.collective_stats),
                          /*setup_seconds=*/0.0,
                          ls.RI.callD - callD_before);
  if (lhs.commu.masF && fsilsTraceEnabled()) {
    const double outer_residual_est =
        (last_i + 1 < err.size()) ? std::abs(err(last_i + 1)) : 0.0;
    std::fprintf(stderr,
                 "[NS_SOLVER] coupled outer FGMRES final: converged=%d outer_iters=%d krylov_dim=%d true_norm=%e true_target=%e outer_est=%e outer_target=%e\n",
                 ls.RI.suc ? 1 : 0,
                 ls.RI.itr,
                 outer_krylov_dim,
                 ls.RI.fNorm,
                 true_eps,
                 outer_residual_est,
                 outer_eps);
  }

  if (ls.RI.suc) {
    Ri = x;
  }
  return ls.RI.suc;
}

} // namespace

/// @brief Modifies: lhs.face[].nS
/// Accumulates boundary coupling norms for the field-A (momentum) components.
void bc_pre(fe_fsi_linear_solver::FSILS_lhsType& lhs, const int mom_ncomp, const int dof, const fsils_int nNo, const fsils_int mynNo)
{
  std::vector<int> shared_face_indices;
  std::vector<double> shared_face_local_norms;
  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];

    if (face.coupledFlag) {
      if (face.sharedFlag) {
        double local_nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          if (Ac < mynNo) {
            for (int i = 0; i < mom_ncomp; i++) {
              local_nS += face.valM(i,a) * face.valM(i,a);
            }
          }
        }
        shared_face_indices.push_back(faIn);
        shared_face_local_norms.push_back(local_nS);

      } else {
        face.nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          for (int i = 0; i < mom_ncomp; i++) {
            face.nS += face.valM(i,a) * face.valM(i,a);
          }
        }
      }
    }
  }

  if (!shared_face_local_norms.empty()) {
    const fe_fsi_linear_solver::CollectiveOps collectives(lhs.commu);
    std::vector<double> shared_face_global_norms = shared_face_local_norms;
    collectives.allreduce_sum(shared_face_local_norms.data(),
                              shared_face_global_norms.data(),
                              static_cast<int>(shared_face_local_norms.size()));
    for (std::size_t i = 0; i < shared_face_indices.size(); ++i) {
      lhs.face[static_cast<std::size_t>(shared_face_indices[i])].nS = shared_face_global_norms[i];
    }
  }

  add_bc_mul::compute_reduced_update_preconditioner_coupling(lhs);
}

/// @brief Store sections of the 'Val' into separate arrays: 'mK', 'mG', 'mD', 'mL'
/// Uses block layout indices to extract blocks from arbitrary positions in the
/// per-node DOF ordering. Exact analytical D is preserved (no transposition to -G^T).
///
/// Modifies: mK, mG, mD, and mL.
void depart(fe_fsi_linear_solver::FSILS_lhsType& lhs,
            const int mom_start, const int mom_ncomp,
            const int con_start, const int con_ncomp,
            const int dof,
            const fsils_int nNo, const fsils_int nnz,
            const Array<double>& Val, Array<double>& mK, Array<double>& mG, Array<double>& mD, Vector<double>& mL)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int nz = 0; nz < nnz; nz++) {
    for (int i = 0; i < mom_ncomp; i++) {
      for (int j = 0; j < mom_ncomp; j++) {
        mK(i*mom_ncomp + j, nz) = Val((mom_start + i)*dof + (mom_start + j), nz);
      }
      mG(i, nz) = Val((mom_start + i)*dof + con_start, nz);
      mD(i, nz) = Val(con_start*dof + (mom_start + i), nz);
    }
    mL(nz) = Val(con_start*dof + con_start, nz);
  }
}

/// @brief Multi-component block extraction.
/// mG(mom_ncomp*con_ncomp, nnz), mD(con_ncomp*mom_ncomp, nnz), mL(con_ncomp*con_ncomp, nnz).
void depart_mc(fe_fsi_linear_solver::FSILS_lhsType& lhs,
               const int mom_start, const int mom_ncomp,
               const int con_start, const int con_ncomp,
               const int dof,
               const fsils_int nNo, const fsils_int nnz,
               const Array<double>& Val, Array<double>& mK, Array<double>& mG,
               Array<double>& mD, Array<double>& mL)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int nz = 0; nz < nnz; nz++) {
    // K block: mom_ncomp × mom_ncomp
    for (int i = 0; i < mom_ncomp; i++) {
      for (int j = 0; j < mom_ncomp; j++) {
        mK(i*mom_ncomp + j, nz) = Val((mom_start + i)*dof + (mom_start + j), nz);
      }
    }
    // G block: mom_ncomp × con_ncomp (maps constraint → momentum)
    for (int i = 0; i < mom_ncomp; i++) {
      for (int j = 0; j < con_ncomp; j++) {
        mG(i*con_ncomp + j, nz) = Val((mom_start + i)*dof + (con_start + j), nz);
      }
    }
    // D block: con_ncomp × mom_ncomp (maps momentum → constraint)
    for (int i = 0; i < con_ncomp; i++) {
      for (int j = 0; j < mom_ncomp; j++) {
        mD(i*mom_ncomp + j, nz) = Val((con_start + i)*dof + (mom_start + j), nz);
      }
    }
    // L block: con_ncomp × con_ncomp
    for (int i = 0; i < con_ncomp; i++) {
      for (int j = 0; j < con_ncomp; j++) {
        mL(i*con_ncomp + j, nz) = Val((con_start + i)*dof + (con_start + j), nz);
      }
    }
  }
}

/// @brief Multi-component constraint fractional-step solver.
/// Called when con_ncomp > 1. Same algorithm as ns_solver but with
/// multi-component constraint arrays (Rc, P, MP are Array/Array3 instead of Vector).
static void ns_solver_mc(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                          fe_fsi_linear_solver::FSILS_lsType& ls,
                          const int dof, const Array<double>& Val, Array<double>& Ri,
                          const int mom_start, const int mom_ncomp,
                          const int con_start, const int con_ncomp)
{
  using namespace consts;
  using namespace fe_fsi_linear_solver;

  const fsils_int nNo = lhs.nNo;
  const fsils_int nnz = lhs.nnz;
  const fsils_int mynNo = lhs.mynNo;
  const HaloExchange halo(lhs);
  const CollectiveOps collectives(lhs.commu);
  const int nsd = mom_ncomp;
  const int iB = ls.RI.mItr;
  const int nB = 2*iB;
  constexpr fsils_int BLOCK_SIZE = 256;

  Vector<double> tmp(nB*nB+nB), tmpG(nB*nB+nB), B(nB), xB(nB), oldxB(nB);
  oldxB = 0.0;
  Array<double> Rm(nsd,nNo), Rmi(nsd,nNo), A(nB,nB);
  Array<double> Rc(con_ncomp,nNo), Rci(con_ncomp,nNo);
  Array3<double> U(nsd,nNo,iB), MU(nsd,nNo,nB);
  Array3<double> Pcon(con_ncomp,nNo,iB), MPcon(con_ncomp,nNo,nB);

  // Extract initial residual components
  #pragma omp parallel for schedule(static)
  for (fsils_int j = 0; j < nNo; j++) {
    for (int i = 0; i < nsd; i++) {
      Rmi(i,j) = Ri(mom_start + i, j);
    }
    for (int i = 0; i < con_ncomp; i++) {
      Rci(i,j) = Ri(con_start + i, j);
    }
  }

  Rm = Rmi;
  Rc = Rci;

  double eps = std::sqrt(std::max(
      0.0,
      combined_block_norm_sq(lhs, nsd, Rm, /*scalar_constraint=*/false, con_ncomp, &Rc, nullptr)));

  ls.RI.iNorm = eps;
  ls.RI.fNorm = eps*eps;

  if (lhs.commu.masF && fsilsTraceEnabled()) {
    fprintf(stderr, "[NS_SOLVER_MC] eps(initial)=%e nsd=%d con_ncomp=%d dof=%d nNo=%lld\n",
            eps, nsd, con_ncomp, dof, (long long)nNo);
  }

  ls.CG.callD = 0.0;
  ls.GM.callD = 0.0;
  ls.RI.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.CG.itr   = 0;
  ls.GM.itr   = 0;
  ls.RI.suc   = false;
  ls.CG.stats.reset();
  ls.GM.stats.reset();
  ls.RI.stats.reset();
  ls.blockschur_stats.reset();
  eps = std::max(ls.RI.absTol, ls.RI.relTol*eps);
  int i_count{0};
  const int min_outer_iters = blockSchurMinOuterIterations();

  auto update_outer_residual = [&](int max_col) {
    #pragma omp parallel for schedule(static)
    for (fsils_int nBlock = 0; nBlock < nNo; nBlock += BLOCK_SIZE) {
      const fsils_int nEnd = std::min(nBlock + BLOCK_SIZE, nNo);
      for (int j = 0; j <= max_col; j++) {
        const double xb_j = xB(j);
        auto MU_j = MU.rslice(j);
        auto MP_j = MPcon.rslice(j);
        if (j == 0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            for (int d = 0; d < nsd; d++) {
              Rm(d,n) = Rmi(d,n) - xb_j * MU_j(d,n);
            }
            for (int k = 0; k < con_ncomp; k++) {
              Rc(k,n) = Rci(k,n) - xb_j * MP_j(k,n);
            }
          }
        } else if (xb_j != 0.0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            for (int d = 0; d < nsd; d++) {
              Rm(d,n) -= xb_j * MU_j(d,n);
            }
            for (int k = 0; k < con_ncomp; k++) {
              Rc(k,n) -= xb_j * MP_j(k,n);
            }
          }
        }
      }
    }

    halo.sync_owned_to_ghost_vector(nsd, Rm, skipPostSolveHaloSync());
    halo.sync_owned_to_ghost_vector(con_ncomp, Rc, skipPostSolveHaloSync());
  };

  auto actual_outer_residual_norm_sq = [&]() {
    return combined_block_norm_sq(lhs, nsd, Rm, /*scalar_constraint=*/false, con_ncomp, &Rc, nullptr);
  };

  // Extract sub-blocks with multi-component constraint
  Array<double> mK(nsd*nsd,nnz), mG(nsd*con_ncomp,nnz), mD(con_ncomp*nsd,nnz), mL(con_ncomp*con_ncomp,nnz);
  depart_mc(lhs, mom_start, mom_ncomp, con_start, con_ncomp, dof, nNo, nnz, Val, mK, mG, mD, mL);
  const auto momentum_system =
      fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, nsd, mK);
  const auto schur_system =
      fe_fsi_linear_solver::distributed_solver_bundles::make_multi_constraint_block_schur_system(
          lhs, nsd, con_ncomp, mK, mD, mG, mL);

  bc_pre(lhs, mom_ncomp, dof, nNo, mynNo);

  DistributedLowRankCorrection explicit_low_rank_correction;
  if (!lhs.reduced_updates.empty() || !lhs.grouped_bordered_field_couplings.empty()) {
    build_explicit_block_corrections(lhs, mom_start, mom_ncomp, con_start, con_ncomp,
                                     explicit_low_rank_correction);
  }

  int iBB{0};

  for (int i = 0; i < ls.RI.mItr; i++) {
    const auto collective_before_outer = lhs.commu.collective_stats;
    int iB = 2*i;
    iBB = 2*i + 1;
    ls.RI.dB = ls.RI.fNorm;
    i_count = i;

    // Solve U = inv(K) * Rm
    auto U_slice = U.rslice(i);
    U_slice = Rm;
    gmres::gmres_v(momentum_system, ls.GM, U_slice);
    halo.sync_owned_to_ghost_vector(nsd, U_slice, skipPostSolveHaloSync());

    // P = D*U (rect: con_ncomp output × mom_ncomp input)
    auto P_slice = Pcon.rslice(i);
    schur_system.D.apply(
        dso::ghost_synced_input(mom_ncomp, U_slice),
        dso::owned_only_output(con_ncomp, P_slice));
    apply_momentum_block_corrections(lhs, explicit_low_rank_correction, U_slice, P_slice);

    // P = Rc - P
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      for (int k = 0; k < con_ncomp; k++) {
        P_slice(k,n) = Rc(k,n) - P_slice(k,n);
      }
    }
    halo.sync_owned_to_ghost_vector(con_ncomp, P_slice, skipPostSolveHaloSync());

    // P = [L - D*H*G]^-1 * P  (multi-component Schur complement)
    bicgs::schur_mc(schur_system, ls.CG, ls.GM, P_slice);
    halo.sync_owned_to_ghost_vector(con_ncomp, P_slice, skipPostSolveHaloSync());

    // MU1 = G*P (rect: mom_ncomp output × con_ncomp input)
    P_slice = Pcon.rslice(i);
    auto MU_iB = MU.rslice(iB);
    schur_system.G.apply(
        dso::ghost_synced_input(con_ncomp, P_slice),
        dso::owned_only_output(mom_ncomp, MU_iB));
    auto MP_iB = MPcon.rslice(iB);
    schur_system.L.apply(
        dso::ghost_synced_input(con_ncomp, Pcon.rslice(i)),
        dso::owned_only_output(con_ncomp, MP_iB));
    apply_constraint_block_corrections(lhs, explicit_low_rank_correction, P_slice, MU_iB, MP_iB);
    halo.sync_owned_to_ghost_vector(nsd, MU_iB, skipPostSolveHaloSync());
    halo.sync_owned_to_ghost_vector(con_ncomp, MP_iB, skipPostSolveHaloSync());

    // MU2 = Rm - G*P
    auto MU_iBB = MU.rslice(iBB);
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      for (int d = 0; d < nsd; d++) {
        MU_iBB(d,n) = Rm(d,n) - MU_iB(d,n);
      }
    }
    halo.sync_owned_to_ghost_vector(nsd, MU_iBB, skipPostSolveHaloSync());

    // U = inv(K) * [Rm - G*P]
    U_slice = MU_iBB;
    gmres::gmres_v(momentum_system, ls.GM, U_slice);
    halo.sync_owned_to_ghost_vector(nsd, U_slice, skipPostSolveHaloSync());

    // MU2 = K*U
    momentum_system.A.apply(
        dso::ghost_synced_input(nsd, U_slice),
        dso::owned_only_output(nsd, MU_iBB));
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, nsd, U_slice, MU_iBB);

    // MP2 = D*U (rect: con_ncomp × mom_ncomp)
    auto MP_iBB = MPcon.rslice(iBB);
    schur_system.D.apply(
        dso::ghost_synced_input(mom_ncomp, U_slice),
        dso::owned_only_output(con_ncomp, MP_iBB));
    apply_momentum_block_corrections(lhs, explicit_low_rank_correction, U_slice, MP_iBB);
    halo.sync_owned_to_ghost_vector(nsd, MU_iBB, skipPostSolveHaloSync());
    halo.sync_owned_to_ghost_vector(con_ncomp, MP_iBB, skipPostSolveHaloSync());

    // GCR inner products
    int c = 0;
    for (int k = iB; k <= iBB; k++) {
      auto MU_k = MU.rslice(k);
      auto MP_k = MPcon.rslice(k);
      for (int j = 0; j <= k; j++) {
        tmp(c) = dot::fsils_nc_dot_v(nsd, mynNo, MU.rslice(j), MU_k) +
                 dot::fsils_nc_dot_v(con_ncomp, mynNo, MPcon.rslice(j), MP_k);
        c = c + 1;
      }
      tmp(c) = dot::fsils_nc_dot_v(nsd, mynNo, MU_k, Rmi) +
               dot::fsils_nc_dot_v(con_ncomp, mynNo, MP_k, Rci);
      c = c + 1;
    }

    if (collectives.distributed()) {
      collectives.allreduce_sum(tmp.data(), tmpG.data(), c);
      tmp = tmpG;
    }

    // Set arrays for Gauss elimination
    c = 0;
    for (int k = iB; k <= iBB; k++) {
      for (int j = 0; j <= k; j++) {
        A(j,k) = tmp(c);
        A(k,j) = tmp(c);
        c = c + 1;
      }
      B(k) = tmp(c);
      c  = c + 1;
    }

    xB = B;

    // Minimize GCR outer residual
    if (ge::ge(nB, iBB+1, A, xB)) {
      oldxB = xB;
    } else {
      ls.blockschur_stats.record_outer_iteration(
          fsils_collective_delta(collective_before_outer, lhs.commu.collective_stats));
      xB = oldxB;
      if (i > 0) {
        iB = iB - 2;
        iBB = iBB - 2;
        i_count = i - 1;
      }
      break;
    }

    double sum = 0.0;
    for (int j = 0; j <= iBB; j++) {
      sum += xB(j) * B(j);
    }
    const double projected_fNorm = std::max(0.0, std::pow(ls.RI.iNorm,2.0) - sum);
    (void)projected_fNorm;
    update_outer_residual(iBB);
    ls.RI.fNorm = actual_outer_residual_norm_sq();
    ls.blockschur_stats.record_outer_iteration(
        fsils_collective_delta(collective_before_outer, lhs.commu.collective_stats));

    const bool outer_converged = ls.RI.fNorm < eps*eps;
    if (outer_converged && (i + 1) >= min_outer_iters) {
      ls.RI.suc = true;
      break;
    }
  } // for i = 0; i < ls.RI.mItr

  if (i_count >= ls.RI.mItr) {
    ls.RI.itr = ls.RI.mItr;
  } else {
    ls.RI.itr = i_count;
  }

  ls.Resc = (ls.RI.fNorm > 0.0)
      ? static_cast<int>(100.0 * std::pow(norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, Rc),2.0) / ls.RI.fNorm)
      : 0;
  ls.Resm = 100 - ls.Resc;

  // Cache-blocked solution reconstruction
  #pragma omp parallel for schedule(static)
  for (fsils_int nBlock = 0; nBlock < nNo; nBlock += BLOCK_SIZE) {
    const fsils_int nEnd = std::min(nBlock + BLOCK_SIZE, nNo);
    for (int i = 0; i <= ls.RI.itr; i++) {
      auto U_i = U.rslice(i);
      auto P_i = Pcon.rslice(i);
      if (i == 0) {
        const double xb_1 = xB(1);
        const double xb_0 = xB(0);
        for (fsils_int n = nBlock; n < nEnd; n++) {
          for (int d = 0; d < nsd; d++) {
            Rmi(d,n) = xb_1 * U_i(d,n);
          }
          for (int k = 0; k < con_ncomp; k++) {
            Rci(k,n) = xb_0 * P_i(k,n);
          }
        }
      } else {
        const int iB = 2*i;
        const int iBB = 2*i + 1;
        const double xb_iBB = xB(iBB);
        const double xb_iB = xB(iB);
        for (fsils_int n = nBlock; n < nEnd; n++) {
          for (int d = 0; d < nsd; d++) {
            Rmi(d,n) += xb_iBB * U_i(d,n);
          }
          for (int k = 0; k < con_ncomp; k++) {
            Rci(k,n) += xb_iB * P_i(k,n);
          }
        }
      }
    }
  }

  ls.RI.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.RI.callD;
  ls.RI.dB = 5.0 * std::log(ls.RI.fNorm / ls.RI.dB);

  if (ls.Resc < 0.0 || ls.Resm < 0.0) {
    ls.Resc = 0;
    ls.Resm = 0;
    ls.RI.dB = 0;
    ls.RI.fNorm = 0.0;
    ls.RI.suc = false;
  }

  ls.RI.fNorm = std::sqrt(ls.RI.fNorm);
  // Write solution back to Ri
  #pragma omp parallel for schedule(static)
  for (fsils_int j = 0; j < nNo; j++) {
    for (int i = 0; i < nsd; i++) {
      Ri(mom_start + i, j) = Rmi(i,j);
    }
    for (int i = 0; i < con_ncomp; i++) {
      Ri(con_start + i, j) = Rci(i,j);
    }
  }
}

/// @brief Fractional-step solver utilizing an exact asymmetric Schur complement.
/// Block layout is read from ls.mom_start/mom_ncomp/con_start/con_ncomp.
/// If ls.mom_ncomp == 0, falls back to legacy behavior (nsd = dof - 1).
///
/// Ri (dof, lhs.nNo)
void ns_solver(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_lsType& ls, const int dof, const Array<double>& Val, Array<double>& Ri)
{
  using namespace consts;
  using namespace fe_fsi_linear_solver;

  #define n_debug_ns_solver
  #ifdef debug_ns_solver
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  double time = fe_fsi_linear_solver::fsils_cpu_t();
  #endif

  const fsils_int nNo = lhs.nNo;
  const fsils_int nnz = lhs.nnz;
  const fsils_int mynNo = lhs.mynNo;
  const HaloExchange halo(lhs);
  const CollectiveOps collectives(lhs.commu);

  // Block layout: use explicit indices if provided, else legacy fallback.
  const int mom_start = (ls.mom_ncomp > 0) ? ls.mom_start : 0;
  const int mom_ncomp = (ls.mom_ncomp > 0) ? ls.mom_ncomp : (dof - 1);
  const int con_start = (ls.mom_ncomp > 0) ? ls.con_start : (dof - 1);
  const int con_ncomp = (ls.mom_ncomp > 0) ? ls.con_ncomp : 1;

  // Multi-component constraint: delegate to dedicated solver.
  if (con_ncomp > 1) {
    ns_solver_mc(lhs, ls, dof, Val, Ri, mom_start, mom_ncomp, con_start, con_ncomp);
    return;
  }

  const int nsd = mom_ncomp;  // alias for SpMV dimension parameter
  const int iB = ls.RI.mItr;
  const int nB = 2*iB;
  constexpr fsils_int BLOCK_SIZE = 256;

  #ifdef debug_ns_solver
  dmsg << "dof: " << dof;
  dmsg << "nsd: " << nsd;
  dmsg << "nNo: " << nNo;
  dmsg << "nnz: " << nnz;
  dmsg << "mynNo: " << mynNo;
  dmsg << "iB: " << iB;
  dmsg << "nB: " << nB;
  Ri.write(msg_prefix+"Ri");
  #endif

  Vector<double> Rc(nNo), Rci(nNo), tmp(nB*nB+nB), tmpG(nB*nB+nB), B(nB), xB(nB), oldxB(nB);
  oldxB = 0.0;
  Array<double> Rm(nsd,nNo), Rmi(nsd,nNo), A(nB,nB), P(nNo,iB), MP(nNo,nB);
  Array3<double> U(nsd,nNo,iB), MU(nsd,nNo,nB);

  static std::uint64_t ns_dump_solve_counter = 0;
  const char* ns_dump_prefix = fsilsNsDumpPrefix();
  const std::uint64_t ns_dump_solve_id =
      (ns_dump_prefix == nullptr) ? 0 : ns_dump_solve_counter++;
  const int ns_dump_max_solves = fsilsNsDumpMaxSolves();
  const bool ns_dump_enabled =
      ns_dump_prefix != nullptr &&
      (ns_dump_max_solves < 0 ||
       ns_dump_solve_id < static_cast<std::uint64_t>(ns_dump_max_solves));
  auto dump_scalar_stage = [&](const char* label, int iter, const Vector<double>& values) {
    if (ns_dump_enabled) {
      dumpScalarVectorByGlobalNode(lhs, ns_dump_prefix, ns_dump_solve_id, iter, label, values);
    }
  };

  #pragma omp parallel for schedule(static)
  for (fsils_int j = 0; j < nNo; j++) {
    for (int i = 0; i < nsd; i++) {
      Rmi(i,j) = Ri(mom_start + i, j);
    }
    Rci(j) = Ri(con_start, j);
  }

  Rm = Rmi;
  Rc = Rci;
  dump_scalar_stage("initial_Rci", -1, Rci);

  double eps = std::sqrt(std::max(
      0.0,
      combined_block_norm_sq(lhs, nsd, Rm, /*scalar_constraint=*/true,
                             /*con_ncomp=*/1, /*constraint_array=*/nullptr, &Rc)));
  const double initial_norm = eps;

  #ifdef debug_ns_solver
  dmsg << "eps (Rm/Rc): " << eps;
  #endif

  ls.RI.iNorm = eps;
  ls.RI.fNorm = eps*eps;

  if (lhs.commu.masF && fsilsTraceEnabled()) {
    fprintf(stderr, "[NS_SOLVER] eps(initial)=%e iNorm=%e fNorm=%e nsd=%d dof=%d nNo=%lld nnz=%lld\n",
            eps, ls.RI.iNorm, ls.RI.fNorm, nsd, dof, (long long)nNo, (long long)nnz);
    fprintf(stderr, "[NS_SOLVER] relTol=%e absTol=%e mItr=%d\n",
            ls.RI.relTol, ls.RI.absTol, ls.RI.mItr);
  }

  // Calling duration
  ls.CG.callD = 0.0;
  ls.GM.callD = 0.0;
  ls.RI.callD = fe_fsi_linear_solver::fsils_cpu_t();

  ls.CG.itr   = 0;
  ls.GM.itr   = 0;
  ls.RI.suc   = false;
  ls.CG.stats.reset();
  ls.GM.stats.reset();
  ls.RI.stats.reset();
  ls.blockschur_stats.reset();
  eps = std::max(ls.RI.absTol, ls.RI.relTol*eps);
  int i_count{0};
  const int min_outer_iters = blockSchurMinOuterIterations();

  auto update_outer_residual = [&](int max_col) {
    #pragma omp parallel for schedule(static)
    for (fsils_int nBlock = 0; nBlock < nNo; nBlock += BLOCK_SIZE) {
      const fsils_int nEnd = std::min(nBlock + BLOCK_SIZE, nNo);

      for (int j = 0; j <= max_col; j++) {
        const double xb_j = xB(j);
        auto MU_j = MU.rslice(j);
        auto MP_j = MP.rcol(j);

        if (j == 0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            for (int d = 0; d < nsd; d++) {
              Rm(d,n) = Rmi(d,n) - xb_j * MU_j(d,n);
            }
            Rc(n) = Rci(n) - xb_j * MP_j(n);
          }
        } else if (xb_j != 0.0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            for (int d = 0; d < nsd; d++) {
              Rm(d,n) -= xb_j * MU_j(d,n);
            }
            Rc(n) -= xb_j * MP_j(n);
          }
        }
      }
    }

    traceSyncOrder(lhs, "before_sync_vector", i_count, mynNo, nNo, ls.RI.fNorm);
    halo.sync_owned_to_ghost_vector(nsd, Rm, skipPostSolveHaloSync());
    traceSyncOrder(lhs, "after_sync_vector", i_count, mynNo, nNo, ls.RI.fNorm);
    traceSyncOrder(lhs, "before_sync_scalar", i_count, mynNo, nNo, ls.RI.fNorm);
    halo.sync_owned_to_ghost_scalar(Rc, skipPostSolveHaloSync());
    traceSyncOrder(lhs, "after_sync_scalar", i_count, mynNo, nNo, ls.RI.fNorm);
  };

  auto actual_outer_residual_norm_sq = [&]() {
    return combined_block_norm_sq(lhs, nsd, Rm, /*scalar_constraint=*/true,
                                  /*con_ncomp=*/1, /*constraint_array=*/nullptr, &Rc);
  };
  #ifdef debug_ns_solver
  dmsg << "eps: " << eps;
  dmsg << "ls.RI.iNorm: " << ls.RI.iNorm;
  dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
  #endif

  // Extract sub-blocks using block layout indices. Exact analytical D is preserved.
  Array<double> mK(nsd*nsd,nnz), mG(nsd,nnz), mD(nsd,nnz);
  Vector<double> mL(nnz);

  depart(lhs, mom_start, mom_ncomp, con_start, con_ncomp, dof, nNo, nnz, Val, mK, mG, mD, mL);
  const auto momentum_system =
      fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, nsd, mK);
  const auto schur_system =
      fe_fsi_linear_solver::distributed_solver_bundles::make_scalar_block_schur_system(
          lhs, nsd, mK, mD, mG, mL);

  if (const char* oracle_path = fsilsOracleBlockCompareFile();
      oracle_path != nullptr) {
    Vector<double> oracle_mode(nNo);
    if (loadOracleScalarMode(lhs, oracle_path, oracle_mode)) {
      Array<double> gl_gp(nsd, nNo), direct_gp(nsd, nNo), diff_gp(nsd, nNo);
      Vector<double> gl_sp(nNo), direct_sp(nNo), diff_sp(nNo);
      gl_gp = 0.0;
      direct_gp = 0.0;
      diff_gp = 0.0;
      gl_sp = 0.0;
      direct_sp = 0.0;
      diff_sp = 0.0;

      schur_system.GL.apply(
          dso::ghost_synced_input(oracle_mode),
          dso::owned_only_output(nsd, gl_gp),
          dso::owned_only_output(gl_sp));

      const auto compute_direct_range = [&](fsils_int i_start, fsils_int i_end) {
        #pragma omp parallel for schedule(static)
        for (fsils_int row = i_start; row < i_end; ++row) {
          for (fsils_int nz = lhs.rowPtr(0, row); nz <= lhs.rowPtr(1, row); ++nz) {
            const fsils_int col = lhs.colPtr(nz);
            const double p_col = oracle_mode(col);
            for (int comp = 0; comp < nsd; ++comp) {
              direct_gp(comp, row) +=
                  Val((mom_start + comp) * dof + con_start, nz) * p_col;
            }
            direct_sp(row) += Val(con_start * dof + con_start, nz) * p_col;
          }
        }
      };

      compute_direct_range(0, lhs.mynNo);

      for (fsils_int row = 0; row < nNo; ++row) {
        diff_sp(row) = direct_sp(row) - gl_sp(row);
        for (int comp = 0; comp < nsd; ++comp) {
          diff_gp(comp, row) = direct_gp(comp, row) - gl_gp(comp, row);
        }
      }

      const double oracle_norm = norm::fsi_ls_norms(mynNo, lhs.commu, oracle_mode);
      const double gl_gp_norm = norm::fsi_ls_normv(nsd, mynNo, lhs.commu, gl_gp);
      const double direct_gp_norm = norm::fsi_ls_normv(nsd, mynNo, lhs.commu, direct_gp);
      const double diff_gp_norm = norm::fsi_ls_normv(nsd, mynNo, lhs.commu, diff_gp);
      const double gl_sp_norm = norm::fsi_ls_norms(mynNo, lhs.commu, gl_sp);
      const double direct_sp_norm = norm::fsi_ls_norms(mynNo, lhs.commu, direct_sp);
      const double diff_sp_norm = norm::fsi_ls_norms(mynNo, lhs.commu, diff_sp);

      if (lhs.commu.masF && fsilsTraceEnabled()) {
        std::fprintf(stderr,
                     "[NS_ORACLE_BLOCK_COMPARE] oracle_l2=%e gl_gp_l2=%e direct_gp_l2=%e diff_gp_l2=%e gl_sp_l2=%e direct_sp_l2=%e diff_sp_l2=%e\n",
                     oracle_norm,
                     gl_gp_norm,
                     direct_gp_norm,
                     diff_gp_norm,
                     gl_sp_norm,
                     direct_sp_norm,
                     diff_sp_norm);
        std::fflush(stderr);
      }
    }
  }

  // Computes lhs.face[].nS for each face.
  bc_pre(lhs, mom_ncomp, dof, nNo, mynNo);

  DistributedLowRankCorrection explicit_low_rank_correction;
  if (!lhs.reduced_updates.empty() || !lhs.grouped_bordered_field_couplings.empty()) {
    build_explicit_block_corrections(lhs, mom_start, mom_ncomp, con_start, /*con_ncomp=*/1,
                                     explicit_low_rank_correction);
  }
  const auto strategy =
      fe_fsi_linear_solver::BlockSchurStrategySelector::select(
          lhs, explicit_low_rank_correction.profile, /*con_ncomp=*/1);
  if (lhs.commu.masF && fsilsTraceEnabled()) {
    std::fprintf(stderr,
                 "[NS_SOLVER] explicit_block_modes=%zu coupled_face_modes=%zu native_face_rank_one_count=%d native_face_duplicate_modes=%zu auto_outer_fgmres=%d force_outer_fgmres=%d\n",
                 static_cast<std::size_t>(explicit_low_rank_correction.profile.projected_mode_count),
                 static_cast<std::size_t>(explicit_low_rank_correction.profile.active_face_corrections),
                 lhs.native_face_rank_one_count,
                 static_cast<std::size_t>(
                     explicit_low_rank_correction.profile.active_duplicate_reduced_corrections),
                 strategy.auto_enable_coupled_outer_fgmres_scalar ? 1 : 0,
                 strategy.force_enable_coupled_outer_fgmres_scalar ? 1 : 0);
  }
  if (strategy.use_coupled_outer_fgmres_scalar) {
    if (lhs.commu.masF && fsilsTraceEnabled()) {
      std::fprintf(stderr,
                   "[NS_SOLVER] attempting coupled outer FGMRES for block-corrected scalar system\n");
    }
    Array<double> coupled_solution = Ri;
    bool coupled_fgmres_succeeded = false;
    bool coupled_fgmres_threw = false;
    const char* coupled_fgmres_error = nullptr;
    try {
      coupled_fgmres_succeeded = ns_solver_coupled_fgmres_scalar(lhs, ls, dof, mom_start, mom_ncomp, con_start,
                                                                 Val, mK, mD, mG, mL,
                                                                 explicit_low_rank_correction,
                                                                 coupled_solution);
    } catch (const std::exception& e) {
      coupled_fgmres_threw = true;
      coupled_fgmres_error = e.what();
    } catch (...) {
      coupled_fgmres_threw = true;
    }

    if (coupled_fgmres_succeeded) {
      if (lhs.commu.masF && fsilsTraceEnabled()) {
        std::fprintf(stderr,
                     "[NS_SOLVER] coupled outer FGMRES converged in %d outer iterations\n",
                     ls.RI.itr);
      }
      Ri = coupled_solution;
      return;
    }

    if (lhs.commu.masF && fsilsTraceEnabled()) {
      if (coupled_fgmres_threw && coupled_fgmres_error != nullptr) {
        std::fprintf(stderr,
                     "[NS_SOLVER] coupled outer FGMRES threw during apply (%s); reverting to legacy BlockSchur loop\n",
                     coupled_fgmres_error);
      } else {
        std::fprintf(stderr,
                     coupled_fgmres_threw
                         ? "[NS_SOLVER] coupled outer FGMRES threw during apply; reverting to legacy BlockSchur loop\n"
                         : "[NS_SOLVER] coupled outer FGMRES did not converge; reverting to legacy BlockSchur loop\n");
      }
    }
    ls.CG.callD = 0.0;
    ls.GM.callD = 0.0;
    ls.RI.callD = fe_fsi_linear_solver::fsils_cpu_t();
    ls.CG.itr = 0;
    ls.GM.itr = 0;
    ls.RI.suc = false;
    ls.CG.stats.reset();
    ls.GM.stats.reset();
    ls.RI.stats.reset();
    ls.blockschur_stats.reset();
    bicgs::reset_schur_cache(ls.CG);
    ls.RI.iNorm = initial_norm;
    ls.RI.fNorm = initial_norm * initial_norm;
    ls.RI.dB = ls.RI.fNorm;
  }

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    #ifdef debug_ns_solver
    dmsg << "faIn: " << faIn << "  face.nS: " << face.nS;
    #endif
    if (lhs.commu.masF && fsilsTraceEnabled()) {
      fprintf(stderr, "[NS_SOLVER] face[%d]: nNo=%d nS=%e coupledFlag=%d incFlag=%d sharedFlag=%d res=%e bGrp=%d\n",
              faIn, face.nNo, face.nS, face.coupledFlag?1:0, face.incFlag?1:0, face.sharedFlag?1:0, face.res, face.bGrp);
    }
   }

  #ifdef debug_ns_solver
  dmsg << "Loop i on ls.RI.mItr ... ";
  #endif
  int iBB{0};

  for (int i = 0; i < ls.RI.mItr; i++) {
    const auto collective_before_outer = lhs.commu.collective_stats;
    #ifdef debug_ns_solver
    auto istr = "_" + std::to_string(i+1);
    dmsg << "---------- i " << i+1 << " ----------";
    #endif

    int iB = 2*i;
    iBB = 2*i + 1;
    ls.RI.dB = ls.RI.fNorm;
    i_count = i;
    #ifdef debug_ns_solver
    dmsg << "iB: " << iB;
    dmsg << "iBB: " << iBB;
    dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
    #endif

    // Solve for U = inv(mK) * Rm
    auto U_slice = U.rslice(i);
    U_slice = Rm;
    gmres::gmres_v(momentum_system, ls.GM, U_slice);
    halo.sync_owned_to_ghost_vector(nsd, U_slice, skipPostSolveHaloSync());
    if (i == 0 && fsilsTraceEnabled()) {
      trace_momentum_block_vector_stats(
          lhs.commu.masF,
          "iter0 momentum_rhs",
          compute_vector_trace_stats(nsd, mynNo, lhs.commu, Rm));
      trace_scalar_block_vector_stats(
          lhs.commu.masF,
          "iter0 constraint_rhs",
          compute_scalar_trace_stats(mynNo, lhs.commu, Rc));
      trace_momentum_block_vector_stats(
          lhs.commu.masF,
          "iter0 momentum_solve_invK_Rm",
          compute_vector_trace_stats(nsd, mynNo, lhs.commu, U_slice));
    }

    // P = D*U (using exact analytical mD)
    auto P_col = P.rcol(i);
    schur_system.D.apply(
        dso::ghost_synced_input(nsd, U_slice),
        dso::owned_only_output(P_col));
    Array<double> P_block;
    copy_scalar_vector_to_array(P_col, P_block);
    apply_momentum_block_corrections(lhs, explicit_low_rank_correction, U_slice, P_block);
    copy_scalar_array_to_vector(P_block, P_col);

    // P = Rc - P
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      P_col(n) = Rc(n) - P_col(n);
    }
    halo.sync_owned_to_ghost_scalar(P_col, skipPostSolveHaloSync());
    dump_scalar_stage("schur_rhs_before_schur", i, P_col);
    Vector<double> iter0_schur_rhs;
    if (i == 0 && fsilsTraceEnabled()) {
      iter0_schur_rhs = P_col;
      trace_scalar_block_vector_stats(
          lhs.commu.masF,
          "iter0 schur_rhs",
          compute_scalar_trace_stats(mynNo, lhs.commu, iter0_schur_rhs));
    }

    // P = [L - D*H*G]^-1 * P
    // VMS FIX: Solved using asymmetric BiCGStab instead of symmetric CGRAD
    bicgs::schur(schur_system, ls.CG, ls.GM, P_col);
    halo.sync_owned_to_ghost_scalar(P_col, skipPostSolveHaloSync());
    dump_scalar_stage("schur_solution_after_schur", i, P_col);
    if (i == 0 && fsilsTraceEnabled()) {
      trace_scalar_block_vector_stats(
          lhs.commu.masF,
          "iter0 schur_solution",
          compute_scalar_trace_stats(mynNo, lhs.commu, P_col),
          dot::fsils_nc_dot_s(mynNo, iter0_schur_rhs, P_col));
    }

    // MU1 = G*P
    #ifdef debug_ns_solver
    dmsg << "i: " << i+1;
    dmsg << "iB: " << iB+1;
    #endif
    P_col = P.rcol(i);
    auto MU_iB = MU.rslice(iB);
    schur_system.G.apply(
        dso::ghost_synced_input(P_col),
        dso::owned_only_output(nsd, MU_iB));
    auto MP_iB = MP.rcol(iB);
    schur_system.L.apply(
        dso::ghost_synced_input(P.rcol(i)),
        dso::owned_only_output(MP_iB));
    dump_scalar_stage("MP_iB_before_constraint_corrections", i, MP_iB);
    copy_scalar_vector_to_array(P_col, P_block);
    Array<double> MP_iB_block;
    copy_scalar_vector_to_array(MP_iB, MP_iB_block);
    apply_constraint_block_corrections(lhs, explicit_low_rank_correction, P_block, MU_iB, MP_iB_block);
    copy_scalar_array_to_vector(MP_iB_block, MP_iB);
    dump_scalar_stage("MP_iB_after_constraint_corrections", i, MP_iB);
    halo.sync_owned_to_ghost_vector(nsd, MU_iB, skipPostSolveHaloSync());
    halo.sync_owned_to_ghost_scalar(MP_iB, skipPostSolveHaloSync());
    if (i == 0 && fsilsTraceEnabled()) {
      trace_momentum_block_vector_stats(
          lhs.commu.masF,
          "iter0 pressure_basis_momentum",
          compute_vector_trace_stats(nsd, mynNo, lhs.commu, MU_iB));
      trace_scalar_block_vector_stats(
          lhs.commu.masF,
          "iter0 pressure_basis_constraint",
          compute_scalar_trace_stats(mynNo, lhs.commu, MP_iB));
    }
    if (i == 0 && fsilsTraceEnabled() && fsilsFullColumnCompareEnabled()) {
      Array<double> zero_momentum(nsd, nNo);
      zero_momentum = 0.0;
      trace_compare_scalar_block_column(lhs, dof, Val, mom_start, mom_ncomp, con_start,
                                        zero_momentum, P_col, MU_iB, MP_iB, "pressure_basis_iter0");
    }

    // MU2 = Rm - G*P
    auto MU_iBB = MU.rslice(iBB);
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      for (int d = 0; d < nsd; d++) {
        MU_iBB(d,n) = Rm(d,n) - MU_iB(d,n);
      }
    }
    halo.sync_owned_to_ghost_vector(nsd, MU_iBB, skipPostSolveHaloSync());

    // U = inv(K) * [Rm - G*P]
    U_slice = MU_iBB;
    gmres::gmres_v(momentum_system, ls.GM, U_slice);
    halo.sync_owned_to_ghost_vector(nsd, U_slice, skipPostSolveHaloSync());

    // MU2 = K*U
    momentum_system.A.apply(
        dso::ghost_synced_input(nsd, U_slice),
        dso::owned_only_output(nsd, MU_iBB));

    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, nsd, U_slice, MU_iBB);

    // MP2 = D*U
    auto MP_iBB = MP.rcol(iBB);
    schur_system.D.apply(
        dso::ghost_synced_input(nsd, U_slice),
        dso::owned_only_output(MP_iBB));
    dump_scalar_stage("MP_iBB_before_momentum_corrections", i, MP_iBB);
    Array<double> MP_iBB_block;
    copy_scalar_vector_to_array(MP_iBB, MP_iBB_block);
    apply_momentum_block_corrections(lhs, explicit_low_rank_correction, U_slice, MP_iBB_block);
    copy_scalar_array_to_vector(MP_iBB_block, MP_iBB);
    dump_scalar_stage("MP_iBB_after_momentum_corrections", i, MP_iBB);
    halo.sync_owned_to_ghost_vector(nsd, MU_iBB, skipPostSolveHaloSync());
    halo.sync_owned_to_ghost_scalar(MP_iBB, skipPostSolveHaloSync());
    if (i == 0 && fsilsTraceEnabled()) {
      trace_momentum_block_vector_stats(
          lhs.commu.masF,
          "iter0 momentum_basis_momentum",
          compute_vector_trace_stats(nsd, mynNo, lhs.commu, MU_iBB));
      trace_scalar_block_vector_stats(
          lhs.commu.masF,
          "iter0 momentum_basis_constraint",
          compute_scalar_trace_stats(mynNo, lhs.commu, MP_iBB));
    }
    if (i == 0 && fsilsTraceEnabled() && fsilsFullColumnCompareEnabled()) {
      Vector<double> zero_constraint(nNo);
      zero_constraint = 0.0;
      trace_compare_scalar_block_column(lhs, dof, Val, mom_start, mom_ncomp, con_start,
                                        U_slice, zero_constraint, MU_iBB, MP_iBB, "momentum_basis_iter0");
    }

    int c = 0;

    for (int k = iB; k <= iBB; k++) {
      auto MU_k = MU.rslice(k);
      auto MP_k = MP.rcol(k);

      for (int j = 0; j <= k; j++) {
        tmp(c) = dot::fsils_nc_dot_v(nsd, mynNo, MU.rslice(j), MU_k) +
                 dot::fsils_nc_dot_s(mynNo, MP.rcol(j), MP_k);
        c = c + 1;
      }

      tmp(c) = dot::fsils_nc_dot_v(nsd, mynNo, MU_k, Rmi) +
               dot::fsils_nc_dot_s(mynNo, MP_k, Rci);
      c = c + 1;
    }

    if (collectives.distributed()) {
      collectives.allreduce_sum(tmp.data(), tmpG.data(), c);
      tmp = tmpG;
    }

    // Set arrays for Gauss elimination
    c = 0;

    for (int k = iB; k <= iBB; k++) {
      for (int j = 0; j <= k; j++) {
        A(j,k) = tmp(c);
        A(k,j) = tmp(c);
        c = c + 1;
      }

      B(k) = tmp(c);
      c  = c + 1;
    }

    xB = B;

    if (lhs.commu.masF && fsilsTraceEnabled()) {
      fprintf(stderr, "[NS_SOLVER] iter=%d Galerkin system (iBB+1=%d):\n", i_count, iBB+1);
      for (int kk = 0; kk <= iBB; kk++) {
        fprintf(stderr, "[NS_SOLVER]   A[%d,:] =", kk);
        for (int jj = 0; jj <= iBB; jj++) {
          fprintf(stderr, " %e", A(jj,kk));
        }
        fprintf(stderr, "  B[%d]=%e\n", kk, B(kk));
      }
    }

    // Minimize GCR outer residual
    if (ge::ge(nB, iBB+1, A, xB)) {
      oldxB = xB;

    } else {
      ls.blockschur_stats.record_outer_iteration(
          fsils_collective_delta(collective_before_outer, lhs.commu.collective_stats));
      xB = oldxB;

      if (i > 0) {
        iB = iB - 2;
        iBB = iBB - 2;
        i_count = i - 1;
      }
      break;
    }

    double sum = 0.0;
    for (int j = 0; j <= iBB; j++) {
      sum += xB(j) * B(j);
    }
    const double projected_fNorm = std::max(0.0, std::pow(ls.RI.iNorm,2.0) - sum);
    update_outer_residual(iBB);
    ls.RI.fNorm = actual_outer_residual_norm_sq();
    ls.blockschur_stats.record_outer_iteration(
        fsils_collective_delta(collective_before_outer, lhs.commu.collective_stats));
    #ifdef debug_ns_solver
    dmsg << "sum: " << sum;
    dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
    #endif

    if (lhs.commu.masF && fsilsTraceEnabled()) {
      fprintf(stderr, "[NS_SOLVER] iter=%d iNorm=%e iNorm^2=%e sum=%e projected_fNorm=%e fNorm=%e\n",
              i_count, ls.RI.iNorm, std::pow(ls.RI.iNorm,2.0), sum, projected_fNorm, ls.RI.fNorm);
      for (int ii = 0; ii <= iBB; ii++) {
        fprintf(stderr, "[NS_SOLVER]   xB(%d)=%e B(%d)=%e product=%e\n",
                ii, xB(ii), ii, B(ii), xB(ii)*B(ii));
      }
      fprintf(stderr, "[NS_SOLVER]   GM.itr=%d GM.fNorm=%e CG.itr=%d CG.fNorm=%e\n",
              ls.GM.itr, ls.GM.fNorm, ls.CG.itr, ls.CG.fNorm);
    }

    const bool outer_converged = ls.RI.fNorm < eps*eps;
    if (outer_converged && (i + 1) >= min_outer_iters) {
      ls.RI.suc = true;
      break;
    }

  } // for i = 0; i < ls.RI.mItr

  if (i_count >= ls.RI.mItr) {
    ls.RI.itr = ls.RI.mItr;
  } else {
    ls.RI.itr = i_count;
  }

  traceSyncOrder(lhs, "before_resc_norm", ls.RI.itr, mynNo, nNo, ls.RI.fNorm);
  ls.Resc = (ls.RI.fNorm > 0.0)
      ? static_cast<int>(100.0 * std::pow(norm::fsi_ls_norms(mynNo, lhs.commu, Rc),2.0) / ls.RI.fNorm)
      : 0;
  traceSyncOrder(lhs, "after_resc_norm", ls.RI.itr, mynNo, nNo, ls.RI.fNorm);
  ls.Resm = 100 - ls.Resc;

  #ifdef debug_ns_solver
  dmsg << "ls.Resc: " << ls.Resc;
  dmsg << "ls.Resm: " << ls.Resm;
  dmsg << "ls.RI.itr: " << ls.RI.itr;
  #endif

  // Cache-blocked solution reconstruction.
  #pragma omp parallel for schedule(static)
  for (fsils_int nBlock = 0; nBlock < nNo; nBlock += BLOCK_SIZE) {
    const fsils_int nEnd = std::min(nBlock + BLOCK_SIZE, nNo);

    for (int i = 0; i <= ls.RI.itr; i++) {
      auto U_i = U.rslice(i);
      auto P_i = P.rcol(i);

      if (i == 0) {
        const double xb_1 = xB(1);
        const double xb_0 = xB(0);
        for (fsils_int n = nBlock; n < nEnd; n++) {
          for (int d = 0; d < nsd; d++) {
            Rmi(d,n) = xb_1 * U_i(d,n);
          }
          Rci(n) = xb_0 * P_i(n);
        }

      } else {
        const int iB = 2*i;
        const int iBB = 2*i + 1;
        const double xb_iBB = xB(iBB);
        const double xb_iB = xB(iB);
        for (fsils_int n = nBlock; n < nEnd; n++) {
          for (int d = 0; d < nsd; d++) {
            Rmi(d,n) += xb_iBB * U_i(d,n);
          }
          Rci(n) += xb_iB * P_i(n);
        }
      }
    }
  }
  dump_scalar_stage("final_Rci_reconstructed", -1, Rci);

  // Set Calling duration.
  ls.RI.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.RI.callD;

  ls.RI.dB = 5.0 * std::log(ls.RI.fNorm / ls.RI.dB);

  if (ls.Resc < 0.0 || ls.Resm < 0.0) {
    ls.Resc = 0;
    ls.Resm = 0;
    ls.RI.dB = 0;
    ls.RI.fNorm = 0.0;
    ls.RI.suc = false;
  }

  ls.RI.fNorm = std::sqrt(ls.RI.fNorm);
  #ifdef debug_ns_solver
  dmsg << "ls.RI.callD: " << ls.RI.callD;
  dmsg << "ls.RI.dB: " << ls.RI.dB;
  dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
  #endif

  #pragma omp parallel for schedule(static)
  for (fsils_int j = 0; j < nNo; j++) {
    for (int i = 0; i < nsd; i++) {
      Ri(mom_start + i, j) = Rmi(i,j);
    }
    Ri(con_start, j) = Rci(j);
  }

  if (lhs.commu.masF) {
    //CALL LOGFILE
  }

  #ifdef debug_ns_solver
  double exec_time = fe_fsi_linear_solver::fsils_cpu_t() - time;
  dmsg << "Execution time: " << exec_time;
  dmsg << "Done";
  #endif
}

} // namespace ns_solver
