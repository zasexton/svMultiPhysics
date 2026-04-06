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
#include "dot.h"
#include "ge.h"
#include "gmres.h"
#include "norm.h"
#include "omp_la.h"
#include "spar_mul.h"

#include "Array3.h"

#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace ns_solver {

using fe_fsi_linear_solver::fsils_int;

namespace {

constexpr int kNativeFaceDuplicateCouplingId = -2;

[[nodiscard]] bool fsilsTraceEnabled() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_TRACE");
  if (env == nullptr) {
    env = std::getenv("SVMP_FSILS_NS_TRACE");
  }
  if (env == nullptr) {
    env = std::getenv("SVMP_FSILS_NS_SOLVER_TRACE");
  }
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

struct ExplicitReducedBlockCorrection {
  double sigma{0.0};
  Array<double> left_momentum;
  Array<double> left_constraint;
  Array<double> right_momentum;
  Array<double> right_constraint;
};

struct ExplicitGroupedBlockCorrection {
  std::vector<double> dense_coeff;
  std::vector<Array<double>> left_momentum_modes;
  std::vector<Array<double>> left_constraint_modes;
  std::vector<Array<double>> right_momentum_modes;
  std::vector<Array<double>> right_constraint_modes;
};

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
  if (lhs.commu.nTasks > 1 && !values.empty()) {
    fe_fsi_linear_solver::fsils_allreduce_sum_in_place(values.data(),
                                                       static_cast<int>(values.size()),
                                                       cm_mod::mpreal,
                                                       lhs.commu);
  }
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
    std::vector<ExplicitReducedBlockCorrection>& reduced,
    std::vector<ExplicitGroupedBlockCorrection>& grouped)
{
  reduced.clear();
  grouped.clear();

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

    ExplicitReducedBlockCorrection corr;
    corr.sigma = update.sigma;
    const auto& left_entries = !update.left_scaled.empty() ? update.left_scaled : update.left;
    const auto& right_entries = !update.right_scaled.empty() ? update.right_scaled : update.right;
    fill_projected_block_vector(lhs, left_entries,
                                mom_start, mom_ncomp, corr.left_momentum);
    fill_projected_block_vector(lhs, left_entries,
                                con_start, con_ncomp, corr.left_constraint);
    fill_projected_block_vector(lhs, right_entries,
                                mom_start, mom_ncomp, corr.right_momentum);
    fill_projected_block_vector(lhs, right_entries,
                                con_start, con_ncomp, corr.right_constraint);
    reduced.push_back(std::move(corr));
  }

  for (const auto& group_data : lhs.grouped_bordered_field_couplings) {
    if (!group_data.active || group_data.modes.empty()) {
      continue;
    }

    const int rank = static_cast<int>(group_data.modes.size());
    if (group_data.aux_matrix.size() != static_cast<std::size_t>(rank * rank)) {
      continue;
    }

    ExplicitGroupedBlockCorrection corr;
    corr.dense_coeff.assign(group_data.aux_matrix.size(), 0.0);
    if (!invert_dense_matrix(rank, group_data.aux_matrix, corr.dense_coeff)) {
      continue;
    }
    for (double& value : corr.dense_coeff) {
      value = -value;
    }

    corr.left_momentum_modes.reserve(group_data.modes.size());
    corr.left_constraint_modes.reserve(group_data.modes.size());
    corr.right_momentum_modes.reserve(group_data.modes.size());
    corr.right_constraint_modes.reserve(group_data.modes.size());

    bool any_left_momentum = false;
    bool any_left_constraint = false;
    bool any_right_momentum = false;
    bool any_right_constraint = false;
    for (const auto& mode : group_data.modes) {
      const auto& left_entries = !mode.left_scaled.empty() ? mode.left_scaled : mode.left;
      const auto& right_entries = !mode.right_scaled.empty() ? mode.right_scaled : mode.right;
      corr.left_momentum_modes.emplace_back();
      corr.left_constraint_modes.emplace_back();
      corr.right_momentum_modes.emplace_back();
      corr.right_constraint_modes.emplace_back();

      any_left_momentum |= fill_projected_block_vector(lhs, left_entries,
                                                       mom_start, mom_ncomp,
                                                       corr.left_momentum_modes.back());
      any_left_constraint |= fill_projected_block_vector(lhs, left_entries,
                                                         con_start, con_ncomp,
                                                         corr.left_constraint_modes.back());
      any_right_momentum |= fill_projected_block_vector(lhs, right_entries,
                                                        mom_start, mom_ncomp,
                                                        corr.right_momentum_modes.back());
      any_right_constraint |= fill_projected_block_vector(lhs, right_entries,
                                                          con_start, con_ncomp,
                                                          corr.right_constraint_modes.back());
    }

    const bool contributes_to_gp_or_lp = any_right_constraint && (any_left_momentum || any_left_constraint);
    const bool contributes_to_d = any_right_momentum && any_left_constraint;
    if (contributes_to_gp_or_lp || contributes_to_d) {
      grouped.push_back(std::move(corr));
    }
  }
}

void apply_constraint_block_corrections(
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const std::vector<ExplicitReducedBlockCorrection>& reduced,
    const std::vector<ExplicitGroupedBlockCorrection>& grouped,
    const Array<double>& in_constraint,
    Array<double>& out_momentum,
    Array<double>& out_constraint)
{
  const int n_reduced = static_cast<int>(reduced.size());
  int total_rhs = n_reduced;
  for (const auto& group : grouped) {
    total_rhs += static_cast<int>(group.right_constraint_modes.size());
  }
  if (total_rhs <= 0) {
    return;
  }

  const int mom_ncomp = out_momentum.nrows();
  const int con_ncomp = out_constraint.nrows();
  const fsils_int nNo = out_momentum.ncols();

  std::vector<double> rhs(static_cast<std::size_t>(total_rhs), 0.0);
  int offset = 0;
  for (const auto& corr : reduced) {
    rhs[static_cast<std::size_t>(offset++)] =
        dense_dense_owned_dot_local(lhs, con_ncomp, corr.right_constraint, in_constraint);
  }
  for (const auto& group : grouped) {
    for (const auto& mode : group.right_constraint_modes) {
      rhs[static_cast<std::size_t>(offset++)] =
          dense_dense_owned_dot_local(lhs, con_ncomp, mode, in_constraint);
    }
  }
  allreduce_sum_in_place(lhs, rhs);

  offset = 0;
  for (const auto& corr : reduced) {
    const double alpha = corr.sigma * rhs[static_cast<std::size_t>(offset++)];
    dense_axpy(mom_ncomp, nNo, corr.left_momentum, alpha, out_momentum);
    dense_axpy(con_ncomp, nNo, corr.left_constraint, alpha, out_constraint);
  }

  for (const auto& group : grouped) {
    const int rank = static_cast<int>(group.right_constraint_modes.size());
    if (rank <= 0 || group.dense_coeff.size() != static_cast<std::size_t>(rank * rank)) {
      offset += rank;
      continue;
    }

    std::vector<double> alpha(static_cast<std::size_t>(rank), 0.0);
    for (int i = 0; i < rank; ++i) {
      double value = 0.0;
      for (int j = 0; j < rank; ++j) {
        value += group.dense_coeff[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                                   static_cast<std::size_t>(j)] *
                 rhs[static_cast<std::size_t>(offset + j)];
      }
      alpha[static_cast<std::size_t>(i)] = value;
    }
    offset += rank;

    for (int i = 0; i < rank; ++i) {
      const double coeff = alpha[static_cast<std::size_t>(i)];
      dense_axpy(mom_ncomp, nNo, group.left_momentum_modes[static_cast<std::size_t>(i)], coeff, out_momentum);
      dense_axpy(con_ncomp, nNo, group.left_constraint_modes[static_cast<std::size_t>(i)], coeff, out_constraint);
    }
  }
}

void apply_momentum_block_corrections(
    fe_fsi_linear_solver::FSILS_lhsType& lhs,
    const std::vector<ExplicitReducedBlockCorrection>& reduced,
    const std::vector<ExplicitGroupedBlockCorrection>& grouped,
    const Array<double>& in_momentum,
    Array<double>& out_constraint)
{
  const int n_reduced = static_cast<int>(reduced.size());
  int total_rhs = n_reduced;
  for (const auto& group : grouped) {
    total_rhs += static_cast<int>(group.right_momentum_modes.size());
  }
  if (total_rhs <= 0) {
    return;
  }

  const int mom_ncomp = in_momentum.nrows();
  const int con_ncomp = out_constraint.nrows();
  const fsils_int nNo = out_constraint.ncols();

  std::vector<double> rhs(static_cast<std::size_t>(total_rhs), 0.0);
  int offset = 0;
  for (const auto& corr : reduced) {
    rhs[static_cast<std::size_t>(offset++)] =
        dense_dense_owned_dot_local(lhs, mom_ncomp, corr.right_momentum, in_momentum);
  }
  for (const auto& group : grouped) {
    for (const auto& mode : group.right_momentum_modes) {
      rhs[static_cast<std::size_t>(offset++)] =
          dense_dense_owned_dot_local(lhs, mom_ncomp, mode, in_momentum);
    }
  }
  allreduce_sum_in_place(lhs, rhs);

  offset = 0;
  for (const auto& corr : reduced) {
    const double alpha = corr.sigma * rhs[static_cast<std::size_t>(offset++)];
    dense_axpy(con_ncomp, nNo, corr.left_constraint, alpha, out_constraint);
  }

  for (const auto& group : grouped) {
    const int rank = static_cast<int>(group.right_momentum_modes.size());
    if (rank <= 0 || group.dense_coeff.size() != static_cast<std::size_t>(rank * rank)) {
      offset += rank;
      continue;
    }

    std::vector<double> alpha(static_cast<std::size_t>(rank), 0.0);
    for (int i = 0; i < rank; ++i) {
      double value = 0.0;
      for (int j = 0; j < rank; ++j) {
        value += group.dense_coeff[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                                   static_cast<std::size_t>(j)] *
                 rhs[static_cast<std::size_t>(offset + j)];
      }
      alpha[static_cast<std::size_t>(i)] = value;
    }
    offset += rank;

    for (int i = 0; i < rank; ++i) {
      dense_axpy(con_ncomp, nNo, group.left_constraint_modes[static_cast<std::size_t>(i)],
                 alpha[static_cast<std::size_t>(i)], out_constraint);
    }
  }
}

[[nodiscard]] bool has_coupled_block_corrections(const fe_fsi_linear_solver::FSILS_lhsType& lhs)
{
  return std::any_of(lhs.face.begin(), lhs.face.end(),
      [](const fe_fsi_linear_solver::FSILS_faceType& face) {
        const int min_supported_nodes = std::max(1, face.dof);
        return face.coupledFlag && std::abs(face.res) > 1e-30 && face.nNo > min_supported_nodes;
      });
}

void apply_full_coupled_operator(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                 int dof,
                                 const Array<double>& Val,
                                 const Array<double>& x,
                                 Array<double>& y)
{
  spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, x, y);
  add_bc_mul::add_bc_mul(lhs, add_bc_mul::BcopType::BCOP_TYPE_ADD, dof, x, y);
}

void apply_full_coupled_outer_operator(fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                       int dof,
                                       const Array<double>& Val,
                                       const Array<double>& x,
                                       Array<double>& y)
{
  apply_full_coupled_operator(lhs, dof, Val, x, y);
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
    const std::vector<ExplicitReducedBlockCorrection>& explicit_reduced_corrections,
    const std::vector<ExplicitGroupedBlockCorrection>& explicit_grouped_corrections,
    Array<double>& Ri)
{
  using namespace fe_fsi_linear_solver;

  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;
  const int pressure_ncomp = 1;
  const int outer_krylov_dim =
      std::max(1, std::min((ls.RI.sD > 0 ? ls.RI.sD : ls.RI.mItr),
                           std::max(ls.RI.mItr, ls.RI.mItr * std::max(1, ls.GM.mItr))));

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
    gmres::gmres_v(lhs, ls.GM, mom_ncomp, mK, mom_sol);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commuv(lhs, mom_ncomp, mom_sol);
    }

    spar_mul::fsils_spar_mul_vs(lhs, lhs.rowPtr, lhs.colPtr, mom_ncomp, mD, mom_sol, con_tmp);
    copy_scalar_vector_to_array(con_tmp, dummy_constraint);
    apply_momentum_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                     mom_sol, dummy_constraint);
    copy_scalar_array_to_vector(dummy_constraint, con_tmp);

    #pragma omp parallel for schedule(static)
    for (fsils_int node = 0; node < nNo; ++node) {
      con_sol(node) = con_rhs(node) - con_tmp(node);
    }

    bicgs::schur_precondition(lhs, ls.CG, mom_ncomp, mK, mD, mG, mL, con_sol);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commus(lhs, con_sol);
    }

    spar_mul::fsils_spar_mul_sv(lhs, lhs.rowPtr, lhs.colPtr, mom_ncomp, mG, con_sol, gp);
    copy_scalar_vector_to_array(con_sol, dummy_constraint);
    Array<double> dummy_lp(pressure_ncomp, nNo);
    dummy_lp = 0.0;
    apply_constraint_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                       dummy_constraint, gp, dummy_lp);

    #pragma omp parallel for schedule(static)
    for (fsils_int node = 0; node < nNo; ++node) {
      for (int comp = 0; comp < mom_ncomp; ++comp) {
        mom_tmp(comp, node) = mom_rhs(comp, node) - gp(comp, node);
      }
    }

    mom_sol = mom_tmp;
    gmres::gmres_v(lhs, ls.GM, mom_ncomp, mK, mom_sol);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commuv(lhs, mom_ncomp, mom_sol);
    }

    assemble_scalar_block_vector(mom_sol, con_sol, dof, mom_start, mom_ncomp, con_start, out_vec);
  };

  auto compute_true_residual = [&](const Array<double>& trial, Array<double>& out_residual) {
    apply_full_coupled_operator(lhs, dof, Val, trial, ax);
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
    apply_full_coupled_outer_operator(lhs, dof, Val, zi, w);
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
  const double momentum_norm = norm::fsi_ls_normv(mom_ncomp, mynNo, lhs.commu, mom_rhs);
  const double constraint_norm = norm::fsi_ls_norms(mynNo, lhs.commu, con_rhs);
  const double final_norm_sq = momentum_norm * momentum_norm + constraint_norm * constraint_norm;

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
          for (int i = 0; i < mom_ncomp; i++) {
            face.nS += face.valM(i,a) * face.valM(i,a);
          }
        }
      }
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

  double eps = std::sqrt(std::pow(norm::fsi_ls_normv(nsd,mynNo,lhs.commu,Rm),2.0) +
                         std::pow(norm::fsi_ls_normv(con_ncomp,mynNo,lhs.commu,Rc),2.0));

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
  bicgs::reset_schur_cache(ls.CG);
  eps = std::max(ls.RI.absTol, ls.RI.relTol*eps);

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
  };

  auto actual_outer_residual_norm_sq = [&]() {
    const double rm_norm = norm::fsi_ls_normv(nsd, mynNo, lhs.commu, Rm);
    const double rc_norm = norm::fsi_ls_normv(con_ncomp, mynNo, lhs.commu, Rc);
    return rm_norm * rm_norm + rc_norm * rc_norm;
  };

  // Extract sub-blocks with multi-component constraint
  Array<double> mK(nsd*nsd,nnz), mG(nsd*con_ncomp,nnz), mD(con_ncomp*nsd,nnz), mL(con_ncomp*con_ncomp,nnz);
  depart_mc(lhs, mom_start, mom_ncomp, con_start, con_ncomp, dof, nNo, nnz, Val, mK, mG, mD, mL);

  bc_pre(lhs, mom_ncomp, dof, nNo, mynNo);

  std::vector<ExplicitReducedBlockCorrection> explicit_reduced_corrections;
  std::vector<ExplicitGroupedBlockCorrection> explicit_grouped_corrections;
  if (!lhs.reduced_updates.empty() || !lhs.grouped_bordered_field_couplings.empty()) {
    build_explicit_block_corrections(lhs, mom_start, mom_ncomp, con_start, con_ncomp,
                                     explicit_reduced_corrections, explicit_grouped_corrections);
  }

  int iBB{0};
  int i_count{0};

  for (int i = 0; i < ls.RI.mItr; i++) {
    const auto collective_before_outer = lhs.commu.collective_stats;
    int iB = 2*i;
    iBB = 2*i + 1;
    ls.RI.dB = ls.RI.fNorm;
    i_count = i;

    // Solve U = inv(K) * Rm
    auto U_slice = U.rslice(i);
    U_slice = Rm;
    gmres::gmres_v(lhs, ls.GM, nsd, mK, U_slice);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commuv(lhs, nsd, U_slice);
    }

    // P = D*U (rect: con_ncomp output × mom_ncomp input)
    auto P_slice = Pcon.rslice(i);
    spar_mul::fsils_spar_mul_rect(lhs, lhs.rowPtr, lhs.colPtr, con_ncomp, mom_ncomp, mD, U_slice, P_slice);
    apply_momentum_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                     U_slice, P_slice);

    // P = Rc - P
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      for (int k = 0; k < con_ncomp; k++) {
        P_slice(k,n) = Rc(k,n) - P_slice(k,n);
      }
    }

    // P = [L - D*H*G]^-1 * P  (multi-component Schur complement)
    bicgs::schur_mc(lhs, ls.CG, ls.GM, nsd, con_ncomp, mK, mD, mG, mL, P_slice);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commuv(lhs, con_ncomp, P_slice);
    }

    // MU1 = G*P (rect: mom_ncomp output × con_ncomp input)
    P_slice = Pcon.rslice(i);
    auto MU_iB = MU.rslice(iB);
    spar_mul::fsils_spar_mul_rect(lhs, lhs.rowPtr, lhs.colPtr, mom_ncomp, con_ncomp, mG, P_slice, MU_iB);
    auto MP_iB = MPcon.rslice(iB);
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, con_ncomp, mL, Pcon.rslice(i), MP_iB);
    apply_constraint_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                       P_slice, MU_iB, MP_iB);

    // MU2 = Rm - G*P
    auto MU_iBB = MU.rslice(iBB);
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      for (int d = 0; d < nsd; d++) {
        MU_iBB(d,n) = Rm(d,n) - MU_iB(d,n);
      }
    }

    // U = inv(K) * [Rm - G*P]
    U_slice = MU_iBB;
    gmres::gmres_v(lhs, ls.GM, nsd, mK, U_slice);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commuv(lhs, nsd, U_slice);
    }

    // MU2 = K*U
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, nsd, mK, U_slice, MU_iBB);
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, nsd, U_slice, MU_iBB);

    // MP2 = D*U (rect: con_ncomp × mom_ncomp)
    auto MP_iBB = MPcon.rslice(iBB);
    spar_mul::fsils_spar_mul_rect(lhs, lhs.rowPtr, lhs.colPtr, con_ncomp, mom_ncomp, mD, U_slice, MP_iBB);
    apply_momentum_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                     U_slice, MP_iBB);

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

    if (lhs.commu.nTasks > 1) {
      fsils_allreduce_sum(tmp.data(), tmpG.data(), c, cm_mod::mpreal, lhs.commu);
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

    if (ls.RI.fNorm < eps*eps) {
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
  bicgs::reset_schur_cache(ls.CG);

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

  #pragma omp parallel for schedule(static)
  for (fsils_int j = 0; j < nNo; j++) {
    for (int i = 0; i < nsd; i++) {
      Rmi(i,j) = Ri(mom_start + i, j);
    }
    Rci(j) = Ri(con_start, j);
  }

  Rm = Rmi;
  Rc = Rci;

  double eps = std::sqrt(std::pow(norm::fsi_ls_normv(nsd,mynNo,lhs.commu,Rm),2.0) +
                         std::pow(norm::fsi_ls_norms(mynNo,lhs.commu,Rc),2.0));
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
  bicgs::reset_schur_cache(ls.CG);
  eps = std::max(ls.RI.absTol, ls.RI.relTol*eps);

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
  };

  auto actual_outer_residual_norm_sq = [&]() {
    const double rm_norm = norm::fsi_ls_normv(nsd, mynNo, lhs.commu, Rm);
    const double rc_norm = norm::fsi_ls_norms(mynNo, lhs.commu, Rc);
    return rm_norm * rm_norm + rc_norm * rc_norm;
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

  // Computes lhs.face[].nS for each face.
  bc_pre(lhs, mom_ncomp, dof, nNo, mynNo);

  std::vector<ExplicitReducedBlockCorrection> explicit_reduced_corrections;
  std::vector<ExplicitGroupedBlockCorrection> explicit_grouped_corrections;
  if (!lhs.reduced_updates.empty() || !lhs.grouped_bordered_field_couplings.empty()) {
    build_explicit_block_corrections(lhs, mom_start, mom_ncomp, con_start, /*con_ncomp=*/1,
                                     explicit_reduced_corrections, explicit_grouped_corrections);
  }

  const bool use_coupled_fgmres_scalar = false &&
      has_coupled_block_corrections(lhs) &&
      explicit_reduced_corrections.empty() &&
      explicit_grouped_corrections.empty();
  if (use_coupled_fgmres_scalar) {
    if (lhs.commu.masF && fsilsTraceEnabled()) {
      std::fprintf(stderr,
                   "[NS_SOLVER] attempting coupled outer FGMRES for native face-coupled scalar system\n");
    }
    Array<double> coupled_solution = Ri;
    bool coupled_fgmres_succeeded = false;
    bool coupled_fgmres_threw = false;
    const char* coupled_fgmres_error = nullptr;
    try {
      coupled_fgmres_succeeded = ns_solver_coupled_fgmres_scalar(lhs, ls, dof, mom_start, mom_ncomp, con_start,
                                                                 Val, mK, mD, mG, mL,
                                                                 explicit_reduced_corrections,
                                                                 explicit_grouped_corrections,
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
      bicgs::reset_schur_cache(ls.CG);
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
  int i_count{0};

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
    gmres::gmres_v(lhs, ls.GM, nsd, mK, U_slice);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commuv(lhs, nsd, U_slice);
    }

    // P = D*U (using exact analytical mD)
    auto P_col = P.rcol(i);
    spar_mul::fsils_spar_mul_vs(lhs, lhs.rowPtr, lhs.colPtr, nsd, mD, U_slice, P_col);
    Array<double> P_block;
    copy_scalar_vector_to_array(P_col, P_block);
    apply_momentum_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                     U_slice, P_block);
    copy_scalar_array_to_vector(P_block, P_col);

    // P = Rc - P
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      P_col(n) = Rc(n) - P_col(n);
    }

    // P = [L - D*H*G]^-1 * P
    // VMS FIX: Solved using asymmetric BiCGStab instead of symmetric CGRAD
    bicgs::schur(lhs, ls.CG, ls.GM, nsd, mK, mD, mG, mL, P_col);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commus(lhs, P_col);
    }

    // MU1 = G*P
    #ifdef debug_ns_solver
    dmsg << "i: " << i+1;
    dmsg << "iB: " << iB+1;
    #endif
    P_col = P.rcol(i);
    auto MU_iB = MU.rslice(iB);
    spar_mul::fsils_spar_mul_sv(lhs, lhs.rowPtr, lhs.colPtr, nsd, mG, P_col, MU_iB);
    auto MP_iB = MP.rcol(iB);
    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, mL, P.rcol(i), MP_iB);
    copy_scalar_vector_to_array(P_col, P_block);
    Array<double> MP_iB_block;
    copy_scalar_vector_to_array(MP_iB, MP_iB_block);
    apply_constraint_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                       P_block, MU_iB, MP_iB_block);
    copy_scalar_array_to_vector(MP_iB_block, MP_iB);

    // MU2 = Rm - G*P
    auto MU_iBB = MU.rslice(iBB);
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      for (int d = 0; d < nsd; d++) {
        MU_iBB(d,n) = Rm(d,n) - MU_iB(d,n);
      }
    }

    // U = inv(K) * [Rm - G*P]
    U_slice = MU_iBB;
    gmres::gmres_v(lhs, ls.GM, nsd, mK, U_slice);
    if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
      fsils_commuv(lhs, nsd, U_slice);
    }

    // MU2 = K*U
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, nsd, mK, U_slice, MU_iBB);

    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, nsd, U_slice, MU_iBB);

    // MP2 = D*U
    auto MP_iBB = MP.rcol(iBB);
    spar_mul::fsils_spar_mul_vs(lhs, lhs.rowPtr, lhs.colPtr, nsd, mD, U_slice, MP_iBB);
    Array<double> MP_iBB_block;
    copy_scalar_vector_to_array(MP_iBB, MP_iBB_block);
    apply_momentum_block_corrections(lhs, explicit_reduced_corrections, explicit_grouped_corrections,
                                     U_slice, MP_iBB_block);
    copy_scalar_array_to_vector(MP_iBB_block, MP_iBB);

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

    if (lhs.commu.nTasks > 1) {
      fsils_allreduce_sum(tmp.data(), tmpG.data(), c, cm_mod::mpreal, lhs.commu);
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

    if (ls.RI.fNorm < eps*eps) {
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
      ? static_cast<int>(100.0 * std::pow(norm::fsi_ls_norms(mynNo, lhs.commu, Rc),2.0) / ls.RI.fNorm)
      : 0;
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
  bicgs::reset_schur_cache(ls.CG);
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
