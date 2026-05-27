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

#include "add_bc_mul.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Math/DenseLinearAlgebra.h"

namespace add_bc_mul {

namespace {

constexpr int kNativeFaceDuplicateCouplingId = -2;

[[nodiscard]] bool native_face_exact_pre_enabled() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_NATIVE_FACE_EXACT_PRE");
  if (env == nullptr) {
    return false;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  return *env != '\0' && *env != '0';
}

[[nodiscard]] bool trace_native_face_exact_pre_enabled() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_TRACE_NATIVE_FACE_EXACT_PRE");
  if (env == nullptr) {
    return false;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  return *env != '\0' && *env != '0';
}

[[nodiscard]] const char* addBcMulDumpPrefix() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_ADD_BC_MUL_DUMP_PREFIX");
  if (env == nullptr) {
    return nullptr;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  return (*env == '\0') ? nullptr : env;
}

[[nodiscard]] int addBcMulDumpMaxCalls() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_ADD_BC_MUL_DUMP_MAX_CALLS");
  if (env == nullptr || *env == '\0') {
    return 80;
  }
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  return (end == env) ? 80 : static_cast<int>(value);
}

[[nodiscard]] const char* bcopTypeName(BcopType op_type) noexcept
{
  switch (op_type) {
    case BcopType::BCOP_TYPE_ADD:
      return "ADD";
    case BcopType::BCOP_TYPE_PRE:
      return "PRE";
    default:
      return "UNKNOWN";
  }
}

struct ReducedEntryKey
{
  fsils_int node = -1;
  int full_component = -1;

  bool operator==(const ReducedEntryKey& other) const noexcept
  {
    return node == other.node && full_component == other.full_component;
  }
};

struct ReducedEntryKeyHash
{
  std::size_t operator()(const ReducedEntryKey& key) const noexcept
  {
    std::size_t seed = std::hash<fsils_int>{}(key.node);
    seed ^= std::hash<int>{}(key.full_component) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    return seed;
  }
};

inline void clear_grouped_reduced_update_preconditioner(FSILS_lhsType& lhs)
{
  lhs.native_face_pc_active_indices.clear();
  lhs.native_face_pc_dense_coeff.clear();
  lhs.reduced_update_pc_active_indices.clear();
  lhs.reduced_update_pc_inner_inv.clear();
  for (auto& group : lhs.grouped_bordered_field_couplings) {
    group.add_dense_coeff.clear();
    group.pre_dense_coeff.clear();
  }
}

inline bool group_has_exact_face_modes(const FSILS_groupedBorderedFieldCouplingType& group) noexcept
{
  if (!group.active) {
    return false;
  }
  const int rank = static_cast<int>(group.modes.size());
  return rank > 0 &&
         group.aux_matrix.size() == static_cast<std::size_t>(rank * rank) &&
         group.left_faces.size() == static_cast<std::size_t>(rank) &&
         group.right_faces.size() == static_cast<std::size_t>(rank);
}

inline bool group_has_exact_add_coeff(const FSILS_groupedBorderedFieldCouplingType& group) noexcept
{
  const int rank = static_cast<int>(group.modes.size());
  return rank > 0 &&
         group.add_dense_coeff.size() == static_cast<std::size_t>(rank * rank);
}

inline bool group_has_exact_pre_coeff(const FSILS_groupedBorderedFieldCouplingType& group) noexcept
{
  const int rank = static_cast<int>(group.modes.size());
  return rank > 0 &&
         group.pre_dense_coeff.size() == static_cast<std::size_t>(rank * rank);
}

inline int entry_local_component(const FSILS_lhsType& lhs,
                                 const FSILS_reducedFieldUpdateType& update,
                                 const FSILS_reducedSparseEntry& entry,
                                 int dof)
{
  if (entry.node < 0) {
    return -1;
  }
  const int system_dof = (lhs.system_dof > 0) ? lhs.system_dof : dof;
  return fe_fsi_linear_solver::fsils_reduced_local_component(update,
                                                              entry.full_component,
                                                              dof,
                                                              system_dof);
}

void dumpReducedAddBcMul(const FSILS_lhsType& lhs,
                         BcopType op_type,
                         int dof,
                         const std::vector<double>& reduced_dots,
                         const std::vector<int>& exact_group_ids)
{
  const char* prefix = addBcMulDumpPrefix();
  if (prefix == nullptr) {
    return;
  }

  static std::uint64_t call_id = 0;
  const std::uint64_t this_call = call_id++;
  const int max_calls = addBcMulDumpMaxCalls();
  if (max_calls >= 0 && this_call >= static_cast<std::uint64_t>(max_calls)) {
    return;
  }

  std::ostringstream path;
  path << prefix << ".call" << this_call << "." << bcopTypeName(op_type)
       << ".rank" << lhs.commu.task << ".txt";
  std::ofstream out(path.str());
  if (!out) {
    return;
  }

  const bool exact_reduced_pre =
      op_type == BcopType::BCOP_TYPE_PRE &&
      !lhs.reduced_update_pc_active_indices.empty() &&
      lhs.reduced_update_pc_inner_inv.size() ==
          lhs.reduced_update_pc_active_indices.size() *
              lhs.reduced_update_pc_active_indices.size();

  out << std::setprecision(17) << std::scientific;
  out << "# task " << lhs.commu.task
      << " nTasks " << lhs.commu.nTasks
      << " call " << this_call
      << " op " << bcopTypeName(op_type)
      << " dof " << dof
      << " reduced_update_count " << lhs.reduced_updates.size()
      << " use_face_cache " << (lhs.use_reduced_face_cache_in_add_bc_mul ? 1 : 0)
      << " exact_reduced_pre " << (exact_reduced_pre ? 1 : 0) << '\n';
  out << "# idx active duplicate handled_exact face_cache grouped_id sigma nS dot generic_coef generic_scale left_nnz right_nnz left_face_nNo right_face_nNo path\n";

  for (std::size_t idx = 0; idx < lhs.reduced_updates.size(); ++idx) {
    const auto& update = lhs.reduced_updates[idx];
    const bool duplicate = update.grouped_coupling_id == kNativeFaceDuplicateCouplingId;
    const bool handled_exact =
        std::find(exact_group_ids.begin(), exact_group_ids.end(), update.grouped_coupling_id) !=
        exact_group_ids.end();
    double generic_coef = update.sigma;
    if (op_type == BcopType::BCOP_TYPE_PRE) {
      generic_coef = -update.sigma / (1.0 + update.sigma * update.nS);
    }
    const double dot = (idx < reduced_dots.size()) ? reduced_dots[idx] : 0.0;
    const bool use_face_path = lhs.use_reduced_face_cache_in_add_bc_mul && update.has_face_cache;
    out << idx << ' '
        << (update.active ? 1 : 0) << ' '
        << (duplicate ? 1 : 0) << ' '
        << (handled_exact ? 1 : 0) << ' '
        << (update.has_face_cache ? 1 : 0) << ' '
        << update.grouped_coupling_id << ' '
        << update.sigma << ' '
        << update.nS << ' '
        << dot << ' '
        << generic_coef << ' '
        << generic_coef * dot << ' '
        << update.left.size() << ' '
        << update.right_owned.size() << ' '
        << update.left_face.nNo << ' '
        << update.right_face.nNo << ' '
        << (use_face_path ? "face" : "sparse") << '\n';
  }

  if (exact_reduced_pre) {
    const int rank = static_cast<int>(lhs.reduced_update_pc_active_indices.size());
    std::vector<double> coarse_rhs(static_cast<std::size_t>(rank), 0.0);
    std::vector<double> coarse_sol(static_cast<std::size_t>(rank), 0.0);
    for (int i = 0; i < rank; ++i) {
      const int update_index =
          lhs.reduced_update_pc_active_indices[static_cast<std::size_t>(i)];
      if (update_index >= 0 && update_index < static_cast<int>(reduced_dots.size())) {
        coarse_rhs[static_cast<std::size_t>(i)] =
            reduced_dots[static_cast<std::size_t>(update_index)];
      }
    }
    for (int i = 0; i < rank; ++i) {
      double value = 0.0;
      for (int j = 0; j < rank; ++j) {
        value += lhs.reduced_update_pc_inner_inv[static_cast<std::size_t>(i) *
                                                     static_cast<std::size_t>(rank) +
                                                 static_cast<std::size_t>(j)] *
                 coarse_rhs[static_cast<std::size_t>(j)];
      }
      coarse_sol[static_cast<std::size_t>(i)] = value;
    }
    out << "# exact_pre rank " << rank << '\n';
    out << "# exact_i update_index rhs coarse_sol effective_scale inner_inv_row\n";
    for (int i = 0; i < rank; ++i) {
      const int update_index =
          lhs.reduced_update_pc_active_indices[static_cast<std::size_t>(i)];
      const double sigma =
          (update_index >= 0 && update_index < static_cast<int>(lhs.reduced_updates.size()))
              ? lhs.reduced_updates[static_cast<std::size_t>(update_index)].sigma
              : 0.0;
      out << i << ' '
          << update_index << ' '
          << coarse_rhs[static_cast<std::size_t>(i)] << ' '
          << coarse_sol[static_cast<std::size_t>(i)] << ' '
          << -sigma * coarse_sol[static_cast<std::size_t>(i)];
      for (int j = 0; j < rank; ++j) {
        out << ' ' << lhs.reduced_update_pc_inner_inv[static_cast<std::size_t>(i) *
                                                          static_cast<std::size_t>(rank) +
                                                      static_cast<std::size_t>(j)];
      }
      out << '\n';
    }
  }
}

inline double sparse_dot_owned(const FSILS_lhsType& lhs,
                               const FSILS_reducedFieldUpdateType& update,
                               const std::vector<FSILS_reducedSparseEntry>& entries,
                               const Array<double>& X,
                               int dof)
{
  double local_dot = 0.0;
  for (const auto& entry : entries) {
    const int comp = entry_local_component(lhs, update, entry, dof);
    if (comp < 0 || comp >= dof) {
      continue;
    }
    local_dot += entry.value * X(comp, entry.node);
  }
  return local_dot;
}

inline double sparse_overlap_dot_owned(
    const std::vector<FSILS_reducedSparseEntry>& left_entries,
    const std::vector<FSILS_reducedSparseEntry>& right_entries)
{
  if (left_entries.empty() || right_entries.empty()) {
    return 0.0;
  }

  std::unordered_map<ReducedEntryKey, double, ReducedEntryKeyHash> left_values;
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

bool invert_dense_matrix(int n,
                         const std::vector<double>& matrix,
                         std::vector<double>& inverse)
{
  if (n <= 0 || matrix.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n)) {
    inverse.clear();
    return false;
  }

  try {
    auto dense_inverse = svmp::FE::math::invert_dense_matrix(
        std::vector<svmp::FE::Real>(matrix.begin(), matrix.end()),
        static_cast<std::size_t>(n),
        "FSILS dense matrix");
    inverse.assign(dense_inverse.begin(), dense_inverse.end());
    return true;
  } catch (const std::exception&) {
    inverse.clear();
    return false;
  }
}

inline void sparse_axpy_full(const FSILS_lhsType& lhs,
                             const FSILS_reducedFieldUpdateType& update,
                             const std::vector<FSILS_reducedSparseEntry>& entries,
                             double scale,
                             int dof,
                             Array<double>& Y)
{
  if (std::abs(scale) <= 1e-30) {
    return;
  }
  for (const auto& entry : entries) {
    const int comp = entry_local_component(lhs, update, entry, dof);
    if (comp < 0 || comp >= dof || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    Y(comp, entry.node) += entry.value * scale;
  }
}

inline double face_dot_owned(const FSILS_lhsType& lhs,
                             const FSILS_faceType& face,
                             const Array<double>& X)
{
  double local_dot = 0.0;
  const int face_dof = std::min(face.dof, X.nrows());
  for (int a = 0; a < face.nNo; ++a) {
    const int node = face.glob(a);
    if (node < 0 || node >= lhs.mynNo) {
      continue;
    }
    for (int i = 0; i < face_dof; ++i) {
      local_dot += face.valM(i, a) * X(i, node);
    }
  }
  return local_dot;
}

inline double face_overlap_owned(const FSILS_lhsType& lhs,
                                 const FSILS_faceType& left,
                                 const FSILS_faceType& right)
{
  double local_dot = 0.0;
  const int dof = std::min(left.dof, right.dof);
  int ia = 0;
  int ib = 0;
  while (ia < left.nNo && ib < right.nNo) {
    const int node_a = left.glob(ia);
    const int node_b = right.glob(ib);
    if (node_a == node_b) {
      if (node_a >= 0 && node_a < lhs.mynNo) {
        for (int i = 0; i < dof; ++i) {
          local_dot += left.valM(i, ia) * right.valM(i, ib);
        }
      }
      ++ia;
      ++ib;
    } else if (node_a < node_b) {
      ++ia;
    } else {
      ++ib;
    }
  }
  return local_dot;
}

inline void face_axpy_full(const FSILS_faceType& face,
                           double scale,
                           Array<double>& Y)
{
  if (std::abs(scale) <= 1e-30) {
    return;
  }
  const int face_dof = std::min(face.dof, Y.nrows());
  for (int a = 0; a < face.nNo; ++a) {
    const int node = face.glob(a);
    if (node < 0 || node >= Y.ncols()) {
      continue;
    }
    for (int i = 0; i < face_dof; ++i) {
      Y(i, node) += face.valM(i, a) * scale;
    }
  }
}

inline void dense_apply(const std::vector<double>& mat,
                        int n,
                        const std::vector<double>& x,
                        std::vector<double>& y)
{
  y.assign(static_cast<std::size_t>(n), 0.0);
  if (n <= 0 || mat.size() != static_cast<std::size_t>(n * n) ||
      x.size() != static_cast<std::size_t>(n)) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += mat[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                 static_cast<std::size_t>(j)] *
             x[static_cast<std::size_t>(j)];
    }
    y[static_cast<std::size_t>(i)] = sum;
  }
}

} // namespace

void compute_reduced_update_preconditioner_coupling(FSILS_lhsType& lhs)
{
  clear_grouped_reduced_update_preconditioner(lhs);

  if (native_face_exact_pre_enabled()) {
    std::vector<int> active_face_indices;
    active_face_indices.reserve(lhs.face.size());
    for (std::size_t idx = 0; idx < lhs.face.size(); ++idx) {
      const auto& face = lhs.face[idx];
      if (!face.coupledFlag || std::abs(face.res) <= 1e-30) {
        continue;
      }
      active_face_indices.push_back(static_cast<int>(idx));
    }

    const int rank = static_cast<int>(active_face_indices.size());
    if (rank > 1) {
      std::vector<double> dense_m(static_cast<std::size_t>(rank) *
                                      static_cast<std::size_t>(rank),
                                  0.0);
      bool valid = true;
      for (int i = 0; i < rank; ++i) {
        const auto& face_i =
            lhs.face[static_cast<std::size_t>(active_face_indices[static_cast<std::size_t>(i)])];
        if (!(std::abs(face_i.res) > 1e-30)) {
          valid = false;
          break;
        }
        for (int j = 0; j < rank; ++j) {
          const auto& face_j = lhs.face[static_cast<std::size_t>(
              active_face_indices[static_cast<std::size_t>(j)])];
          dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                  static_cast<std::size_t>(j)] =
              face_overlap_owned(lhs, face_j, face_i);
        }
        dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                static_cast<std::size_t>(i)] +=
            1.0 / face_i.res;
      }

      if (valid) {
        if (lhs.commu.nTasks > 1) {
          fsils_allreduce_sum_in_place(dense_m.data(),
                                       static_cast<int>(dense_m.size()),
                                       cm_mod::mpreal,
                                       lhs.commu);
        }

        std::vector<double> dense_inv;
        if (invert_dense_matrix(rank, dense_m, dense_inv)) {
          if (lhs.commu.masF && trace_native_face_exact_pre_enabled()) {
            std::fprintf(stderr, "[NATIVE_FACE_EXACT_PRE] rank=%d\n", rank);
            for (int i = 0; i < rank; ++i) {
              std::fprintf(stderr, "[NATIVE_FACE_EXACT_PRE] row=%d", i);
              for (int j = 0; j < rank; ++j) {
                std::fprintf(stderr,
                             " dense_m[%d,%d]=%.17e dense_inv[%d,%d]=%.17e",
                             i,
                             j,
                             dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                                     static_cast<std::size_t>(j)],
                             i,
                             j,
                             dense_inv[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                                       static_cast<std::size_t>(j)]);
              }
              std::fprintf(stderr, "\n");
            }
            std::fflush(stderr);
          }
          for (double& value : dense_inv) {
            value = -value;
          }
          lhs.native_face_pc_active_indices = std::move(active_face_indices);
          lhs.native_face_pc_dense_coeff = std::move(dense_inv);
        }
      }
    }
  }

  for (auto& group : lhs.grouped_bordered_field_couplings) {
    if (!group_has_exact_face_modes(group)) {
      continue;
    }

    const int rank = static_cast<int>(group.modes.size());
    std::vector<double> add_dense_coeff;
    if (!invert_dense_matrix(rank, group.aux_matrix, add_dense_coeff)) {
      continue;
    }
    for (double& value : add_dense_coeff) {
      value = -value;
    }

    std::vector<double> overlap(static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank), 0.0);
    for (int i = 0; i < rank; ++i) {
      for (int j = 0; j < rank; ++j) {
        overlap[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                static_cast<std::size_t>(j)] =
            face_overlap_owned(lhs,
                               group.right_faces[static_cast<std::size_t>(i)],
                               group.left_faces[static_cast<std::size_t>(j)]);
      }
    }
    if (lhs.commu.nTasks > 1) {
      fsils_allreduce_sum_in_place(overlap.data(),
                                   static_cast<int>(overlap.size()),
                                   cm_mod::mpreal,
                                   lhs.commu);
    }

    std::vector<double> pre_system = group.aux_matrix;
    for (std::size_t idx = 0; idx < pre_system.size(); ++idx) {
      pre_system[idx] -= overlap[idx];
    }

    group.add_dense_coeff = std::move(add_dense_coeff);

    std::vector<double> pre_dense_coeff;
    if (invert_dense_matrix(rank, pre_system, pre_dense_coeff)) {
      group.pre_dense_coeff = std::move(pre_dense_coeff);
    }
  }

  if (lhs.reduced_updates.empty()) {
    return;
  }

  std::vector<int> active_indices;
  active_indices.reserve(lhs.reduced_updates.size());

  for (auto& update : lhs.reduced_updates) {
    if (!update.active) {
      update.nS = 0.0;
      continue;
    }
    if (update.grouped_coupling_id == kNativeFaceDuplicateCouplingId) {
      update.nS = 0.0;
      continue;
    }

    const auto& left_entries =
        update.left_scaled_owned.empty() ? update.left_owned : update.left_scaled_owned;
    const auto& right_entries =
        update.right_scaled_owned.empty() ? update.right_owned : update.right_scaled_owned;
    double local_nS = sparse_overlap_dot_owned(left_entries, right_entries);
    double global_nS = local_nS;
    if (lhs.commu.nTasks > 1) {
      fsils_allreduce_sum(&local_nS, &global_nS, 1, cm_mod::mpreal, lhs.commu);
    }
    update.nS = global_nS;

    if (std::abs(update.sigma) > 1e-30) {
      active_indices.push_back(static_cast<int>(&update - lhs.reduced_updates.data()));
    }
  }

  const int rank = static_cast<int>(active_indices.size());
  if (rank <= 0) {
    return;
  }

  std::vector<double> dense_m(static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank), 0.0);

  for (int i = 0; i < rank; ++i) {
    const auto& update_i = lhs.reduced_updates[static_cast<std::size_t>(active_indices[static_cast<std::size_t>(i)])];
    const auto& right_i =
        update_i.right_scaled_owned.empty() ? update_i.right_owned : update_i.right_scaled_owned;
    for (int j = 0; j < rank; ++j) {
      const auto& update_j =
          lhs.reduced_updates[static_cast<std::size_t>(active_indices[static_cast<std::size_t>(j)])];
      const auto& left_j =
          update_j.left_scaled_owned.empty() ? update_j.left_owned : update_j.left_scaled_owned;
      dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
              static_cast<std::size_t>(j)] +=
          update_j.sigma * sparse_overlap_dot_owned(left_j, right_i);
    }
  }

  if (lhs.commu.nTasks > 1) {
    fsils_allreduce_sum_in_place(dense_m.data(),
                                 static_cast<int>(dense_m.size()),
                                 cm_mod::mpreal,
                                 lhs.commu);
  }
  for (int i = 0; i < rank; ++i) {
    dense_m[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
            static_cast<std::size_t>(i)] += 1.0;
  }

  std::vector<double> dense_inv;
  if (invert_dense_matrix(rank, dense_m, dense_inv)) {
    lhs.reduced_update_pc_active_indices = std::move(active_indices);
    lhs.reduced_update_pc_inner_inv = std::move(dense_inv);
  }
}

/// @brief The contribution of coupled BCs is added to the matrix-vector
/// product operation. Depending on the type of operation (adding the
/// contribution or computing the PC contribution) different
/// coefficients are used.
///
/// For reference, see 
/// Moghadam et al. 2013 eq. 27 (https://doi.org/10.1016/j.jcp.2012.07.035) and
/// Moghadam et al. 2013b (https://doi.org/10.1007/s00466-013-0868-1).
///
/// Reproduces code in ADDBCMUL.f.
/// @param lhs The left-hand side of the linear system. 0D resistance is stored in the face(i).res field.
/// @param op_Type The type of operation (addition or PC contribution)
/// @param dof The number of degrees of freedom.
/// @param X The input vector.
/// @param Y The current matrix-vector product (Y = K*X), to which we add K^BC * X = res * v * v^T * X.
/// The expression is slightly different if preconditioning.
void add_bc_mul(FSILS_lhsType& lhs, const BcopType op_Type, const int dof, const Array<double>& X, Array<double>& Y)
{
  thread_local std::vector<int> shared_face_indices;
  thread_local std::vector<double> shared_face_dot;
  thread_local std::vector<int> exact_group_ids;
  thread_local std::vector<double> native_face_rhs;
  thread_local std::vector<double> native_face_alpha;

  shared_face_indices.clear();
  shared_face_dot.clear();
  shared_face_indices.reserve(static_cast<size_t>(lhs.nFaces));
  shared_face_dot.reserve(static_cast<size_t>(lhs.nFaces));
  exact_group_ids.clear();
  exact_group_ids.reserve(lhs.grouped_bordered_field_couplings.size());
  native_face_rhs.clear();
  native_face_alpha.clear();

  auto grouped_update_handled_exactly = [&](int grouped_coupling_id) {
    return std::find(exact_group_ids.begin(), exact_group_ids.end(), grouped_coupling_id) !=
           exact_group_ids.end();
  };

  const bool use_exact_native_face_pre =
      op_Type == BcopType::BCOP_TYPE_PRE &&
      lhs.native_face_pc_active_indices.size() > 1 &&
      lhs.native_face_pc_dense_coeff.size() ==
          lhs.native_face_pc_active_indices.size() *
              lhs.native_face_pc_active_indices.size();
  auto native_face_handled_exactly = [&](int face_index) {
    return std::find(lhs.native_face_pc_active_indices.begin(),
                     lhs.native_face_pc_active_indices.end(),
                     face_index) != lhs.native_face_pc_active_indices.end();
  };

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    const int face_dof = std::min(face.dof, dof);

    if (face.coupledFlag) {
      if (use_exact_native_face_pre && native_face_handled_exactly(faIn)) {
        continue;
      }

      double coef;
      if (op_Type == BcopType::BCOP_TYPE_ADD) {
        coef = face.res;
      } else {
        coef = -face.res / (1.0 + (face.res * face.nS));
      }

      // If face is shared across procs: compute dot product directly
      // over boundary nodes and use MPI_Allreduce, avoiding a full-mesh
      // temporary vector allocation.
      if (face.sharedFlag) {
        double local_S = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          // Only sum owned nodes (Ac < mynNo) to avoid double-counting
          if (Ac < lhs.mynNo) {
            for (int i = 0; i < face_dof; i++) {
              local_S += face.valM(i,a) * X(i,Ac);
            }
          }
        }
        shared_face_indices.push_back(faIn);
        shared_face_dot.push_back(local_S);

      }
      // If face is not shared across procs
      else {
        // Computing S = coef * v^T * X
        double S = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < face_dof; i++) {
            S = S + face.valM(i,a)*X(i,Ac);
          }
        }
        S = coef * S;

        // Computing Y = Y + v * S
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < face_dof; i++) {
            Y(i,Ac) = Y(i,Ac) + face.valM(i,a)*S;
          }
        }
      }
    }
  }

  if (use_exact_native_face_pre) {
    const int rank = static_cast<int>(lhs.native_face_pc_active_indices.size());
    native_face_rhs.assign(static_cast<std::size_t>(rank), 0.0);
    native_face_alpha.assign(static_cast<std::size_t>(rank), 0.0);

    for (int i = 0; i < rank; ++i) {
      const auto& face = lhs.face[static_cast<std::size_t>(
          lhs.native_face_pc_active_indices[static_cast<std::size_t>(i)])];
      native_face_rhs[static_cast<std::size_t>(i)] = face_dot_owned(lhs, face, X);
    }

    if (lhs.commu.nTasks > 1) {
      fsils_allreduce_sum_in_place(native_face_rhs.data(),
                                   rank,
                                   cm_mod::mpreal,
                                   lhs.commu);
    }

    dense_apply(lhs.native_face_pc_dense_coeff, rank, native_face_rhs, native_face_alpha);

    for (int i = 0; i < rank; ++i) {
      const auto& face = lhs.face[static_cast<std::size_t>(
          lhs.native_face_pc_active_indices[static_cast<std::size_t>(i)])];
      face_axpy_full(face, native_face_alpha[static_cast<std::size_t>(i)], Y);
    }
  }

  if (!shared_face_indices.empty()) {
    if (lhs.commu.nTasks > 1) {
      const int count = static_cast<int>(shared_face_dot.size());
      fsils_allreduce_sum_in_place(shared_face_dot.data(), count, cm_mod::mpreal, lhs.commu);
    }

    for (size_t idx = 0; idx < shared_face_indices.size(); ++idx) {
      auto& face = lhs.face[shared_face_indices[idx]];
      const int face_dof = std::min(face.dof, dof);

      double coef;
      if (op_Type == BcopType::BCOP_TYPE_ADD) {
        coef = face.res;
      } else {
        coef = -face.res / (1.0 + (face.res * face.nS));
      }

      const double S = shared_face_dot[idx] * coef;

      // Computing Y = Y + valM * S
      for (int a = 0; a < face.nNo; a++) {
        const int Ac = face.glob(a);
        for (int i = 0; i < face_dof; i++) {
          Y(i,Ac) = Y(i,Ac) + face.valM(i,a) * S;
        }
      }
    }
  }

  for (auto& group : lhs.grouped_bordered_field_couplings) {
    if (!group.active || group.grouped_coupling_id < 0 || !group_has_exact_face_modes(group)) {
      continue;
    }

    const int rank = static_cast<int>(group.modes.size());
    const std::vector<double>* dense_coeff = nullptr;
    if (op_Type == BcopType::BCOP_TYPE_ADD && group_has_exact_add_coeff(group)) {
      dense_coeff = &group.add_dense_coeff;
    } else if (op_Type == BcopType::BCOP_TYPE_PRE && group_has_exact_pre_coeff(group)) {
      dense_coeff = &group.pre_dense_coeff;
    }
    if (dense_coeff == nullptr ||
        dense_coeff->size() != static_cast<std::size_t>(rank * rank)) {
      continue;
    }

    std::vector<double> rhs(static_cast<std::size_t>(rank), 0.0);
    for (int i = 0; i < rank; ++i) {
      rhs[static_cast<std::size_t>(i)] =
          face_dot_owned(lhs, group.right_faces[static_cast<std::size_t>(i)], X);
    }
    if (lhs.commu.nTasks > 1) {
      fsils_allreduce_sum_in_place(rhs.data(),
                                   static_cast<int>(rhs.size()),
                                   cm_mod::mpreal,
                                   lhs.commu);
    }

    std::vector<double> alpha(static_cast<std::size_t>(rank), 0.0);
    for (int i = 0; i < rank; ++i) {
      double value = 0.0;
      for (int j = 0; j < rank; ++j) {
        value += (*dense_coeff)[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                                static_cast<std::size_t>(j)] *
                 rhs[static_cast<std::size_t>(j)];
      }
      alpha[static_cast<std::size_t>(i)] = value;
    }

    for (int i = 0; i < rank; ++i) {
      face_axpy_full(group.left_faces[static_cast<std::size_t>(i)],
                     alpha[static_cast<std::size_t>(i)],
                     Y);
    }
    exact_group_ids.push_back(group.grouped_coupling_id);
  }

  if (!lhs.reduced_updates.empty()) {
    std::vector<double> reduced_dots(lhs.reduced_updates.size(), 0.0);
    for (std::size_t idx = 0; idx < lhs.reduced_updates.size(); ++idx) {
      const auto& update = lhs.reduced_updates[idx];
      if (!update.active || update.grouped_coupling_id == kNativeFaceDuplicateCouplingId ||
          grouped_update_handled_exactly(update.grouped_coupling_id)) {
        continue;
      }
      if (lhs.use_reduced_face_cache_in_add_bc_mul &&
          update.has_face_cache) {
        reduced_dots[idx] = face_dot_owned(lhs, update.right_face, X);
      } else {
        const auto& right_entries =
            update.right_scaled_owned.empty() ? update.right_owned : update.right_scaled_owned;
        reduced_dots[idx] = sparse_dot_owned(lhs, update, right_entries, X, dof);
      }
    }

    if (lhs.commu.nTasks > 1 && !reduced_dots.empty()) {
      fsils_allreduce_sum_in_place(reduced_dots.data(),
                                   static_cast<int>(reduced_dots.size()),
                                   cm_mod::mpreal,
                                   lhs.commu);
    }
    dumpReducedAddBcMul(lhs, op_Type, dof, reduced_dots, exact_group_ids);

    // Apply the reduced modes as one grouped coarse correction instead of
    // independent scalar Sherman-Morrison updates. This preserves the exact
    // condensed low-rank coupling when multiple outlet modes interact.
    if (op_Type == BcopType::BCOP_TYPE_PRE &&
        !lhs.reduced_update_pc_active_indices.empty() &&
        lhs.reduced_update_pc_inner_inv.size() ==
            lhs.reduced_update_pc_active_indices.size() *
                lhs.reduced_update_pc_active_indices.size()) {
      const int rank = static_cast<int>(lhs.reduced_update_pc_active_indices.size());
      std::vector<double> coarse_rhs(static_cast<std::size_t>(rank), 0.0);
      std::vector<double> coarse_sol(static_cast<std::size_t>(rank), 0.0);
      for (int i = 0; i < rank; ++i) {
        const int update_index =
            lhs.reduced_update_pc_active_indices[static_cast<std::size_t>(i)];
        coarse_rhs[static_cast<std::size_t>(i)] =
            reduced_dots[static_cast<std::size_t>(update_index)];
      }
      for (int i = 0; i < rank; ++i) {
        double value = 0.0;
        for (int j = 0; j < rank; ++j) {
          value += lhs.reduced_update_pc_inner_inv[static_cast<std::size_t>(i) *
                                                       static_cast<std::size_t>(rank) +
                                                   static_cast<std::size_t>(j)] *
                   coarse_rhs[static_cast<std::size_t>(j)];
        }
        coarse_sol[static_cast<std::size_t>(i)] = value;
      }

      for (int i = 0; i < rank; ++i) {
        const auto& update = lhs.reduced_updates[static_cast<std::size_t>(
            lhs.reduced_update_pc_active_indices[static_cast<std::size_t>(i)])];
        if (lhs.use_reduced_face_cache_in_add_bc_mul &&
            update.has_face_cache) {
          face_axpy_full(update.left_face,
                         -update.sigma * coarse_sol[static_cast<std::size_t>(i)],
                         Y);
        } else {
          const auto& left_entries =
              update.left_scaled.empty() ? update.left : update.left_scaled;
          sparse_axpy_full(lhs,
                           update,
                           left_entries,
                           -update.sigma * coarse_sol[static_cast<std::size_t>(i)],
                           dof,
                           Y);
        }
      }
      return;
    }

    for (std::size_t idx = 0; idx < lhs.reduced_updates.size(); ++idx) {
      const auto& update = lhs.reduced_updates[idx];
      if (!update.active || update.grouped_coupling_id == kNativeFaceDuplicateCouplingId ||
          grouped_update_handled_exactly(update.grouped_coupling_id)) {
        continue;
      }

      double coef = update.sigma;
      if (op_Type == BcopType::BCOP_TYPE_PRE) {
        coef = -update.sigma / (1.0 + update.sigma * update.nS);
      }

      if (lhs.use_reduced_face_cache_in_add_bc_mul &&
          update.has_face_cache) {
        face_axpy_full(update.left_face, coef * reduced_dots[idx], Y);
      } else {
        const auto& left_entries =
            update.left_scaled.empty() ? update.left : update.left_scaled;
        sparse_axpy_full(lhs, update, left_entries, coef * reduced_dots[idx], dof, Y);
      }
    }
  }

}

};
