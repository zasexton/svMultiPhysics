#include "distributed_low_rank_correction.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

namespace fe_fsi_linear_solver::distributed_low_rank_correction {

namespace {

constexpr int kNativeFaceDuplicateCouplingId = -2;

[[nodiscard]] bool correctionTraceEnabled() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_TRACE_EXPLICIT_BLOCK_CORRECTIONS");
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

[[nodiscard]] int correctionTraceLimit() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_TRACE_EXPLICIT_BLOCK_CORRECTIONS_LIMIT");
  if (env == nullptr) {
    return 6;
  }
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  if (end == env || value <= 0) {
    return 6;
  }
  if (value > 1024) {
    return 1024;
  }
  return static_cast<int>(value);
}

void maybeTraceCorrectionCoefficients(FSILS_lhsType& lhs,
                                      const char* label,
                                      const std::vector<double>& rhs,
                                      const std::vector<double>* alpha = nullptr)
{
  if (!lhs.commu.masF || !correctionTraceEnabled()) {
    return;
  }

  static int trace_count = 0;
  if (trace_count >= correctionTraceLimit()) {
    return;
  }
  ++trace_count;

  std::fprintf(stderr,
               "[LOW_RANK_CORR] call=%d label=%s rhs_count=%zu",
               trace_count,
               label,
               rhs.size());
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    std::fprintf(stderr, " rhs[%zu]=%.17e", i, rhs[i]);
  }
  if (alpha != nullptr) {
    for (std::size_t i = 0; i < alpha->size(); ++i) {
      std::fprintf(stderr, " alpha[%zu]=%.17e", i, (*alpha)[i]);
    }
  }
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

[[nodiscard]] int reduced_local_component(const FSILS_lhsType& lhs,
                                          const FSILS_reducedFieldUpdateType& update,
                                          int full_component,
                                          int current_dof)
{
  const int system_dof = (lhs.system_dof > 0) ? lhs.system_dof : current_dof;
  return fsils_reduced_local_component(update, full_component, current_dof, system_dof);
}

bool fill_projected_block_vector(
    const FSILS_lhsType& lhs,
    const std::vector<FSILS_reducedSparseEntry>& entries,
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

double dense_dense_owned_dot_local(const FSILS_lhsType& lhs,
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

void allreduce_sum_in_place(FSILS_lhsType& lhs,
                            std::vector<double>& values)
{
  if (lhs.commu.nTasks > 1 && !values.empty()) {
    fsils_allreduce_sum_in_place(values.data(),
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

bool entries_touch_constraint_block(
    const std::vector<FSILS_reducedSparseEntry>& entries,
    int con_start,
    int con_ncomp)
{
  for (const auto& entry : entries) {
    if (entry.node < 0 || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    if (entry.full_component >= con_start &&
        entry.full_component < (con_start + con_ncomp)) {
      return true;
    }
  }
  return false;
}

bool reduced_updates_touch_constraint_block(const FSILS_lhsType& lhs,
                                            int con_start,
                                            int con_ncomp)
{
  bool local_touch = false;
  for (const auto& update : lhs.reduced_updates) {
    if (!update.active || std::abs(update.sigma) <= 1e-30) {
      continue;
    }
    const auto& left_entries = !update.left_scaled.empty() ? update.left_scaled : update.left;
    const auto& right_entries = !update.right_scaled.empty() ? update.right_scaled : update.right;
    if (entries_touch_constraint_block(left_entries, con_start, con_ncomp) ||
        entries_touch_constraint_block(right_entries, con_start, con_ncomp)) {
      local_touch = true;
      break;
    }
  }

  int local_touch_int = local_touch ? 1 : 0;
  int global_touch_int = local_touch_int;
  if (lhs.commu.nTasks > 1) {
    fsils_allreduce_sum(&local_touch_int,
                        &global_touch_int,
                        1,
                        cm_mod::mpint,
                        const_cast<FSILS_commuType&>(lhs.commu));
  }
  return global_touch_int != 0;
}

bool grouped_bordered_touch_constraint_block(const FSILS_lhsType& lhs,
                                             int con_start,
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
      if (entries_touch_constraint_block(left_entries, con_start, con_ncomp) ||
          entries_touch_constraint_block(right_entries, con_start, con_ncomp)) {
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
  if (lhs.commu.nTasks > 1) {
    fsils_allreduce_sum(&local_touch_int,
                        &global_touch_int,
                        1,
                        cm_mod::mpint,
                        const_cast<FSILS_commuType&>(lhs.commu));
  }
  return global_touch_int != 0;
}

bool sparse_entry_sets_match(
    const std::vector<FSILS_reducedSparseEntry>& left_entries,
    const std::vector<FSILS_reducedSparseEntry>& right_entries,
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

bool reduced_update_has_distinct_left_right(const FSILS_reducedFieldUpdateType& update)
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

const std::vector<FSILS_reducedSparseEntry>& projected_left_entries(
    const FSILS_reducedFieldUpdateType& update)
{
  // The exact operator contracts against owned rows on the right, but it
  // scatters the correction onto the full local view on the left.
  return !update.left_scaled.empty()     ? update.left_scaled
      : !update.left.empty()             ? update.left
      : !update.left_scaled_owned.empty() ? update.left_scaled_owned
                                         : update.left_owned;
}

const std::vector<FSILS_reducedSparseEntry>& projected_right_entries(
    const FSILS_reducedFieldUpdateType& update)
{
  return !update.right_scaled_owned.empty() ? update.right_scaled_owned
      : !update.right_owned.empty()         ? update.right_owned
      : !update.right_scaled.empty()        ? update.right_scaled
                                            : update.right;
}

}  // namespace

Profile inspect(const FSILS_lhsType& lhs,
                int mom_start,
                int mom_ncomp,
                int con_start,
                int con_ncomp)
{
  Profile profile;
  (void)mom_start;
  (void)mom_ncomp;
  profile.distributed = lhs.commu.nTasks > 1;
  profile.has_grouped_bordered = !lhs.grouped_bordered_field_couplings.empty();

  int local_active_face_corrections = 0;
  for (const auto& face : lhs.face) {
    if (face.coupledFlag && std::abs(face.res) > 1e-30 && face.nNo > 0) {
      ++local_active_face_corrections;
    }
  }

  int local_active_reduced_corrections = 0;
  int local_active_duplicate_reduced_corrections = 0;
  bool local_has_distinct_multi_reduced_corrections = false;
  for (const auto& update : lhs.reduced_updates) {
    if (!update.active || std::abs(update.sigma) <= 1e-30) {
      continue;
    }
    ++local_active_reduced_corrections;
    if (update.grouped_coupling_id == kNativeFaceDuplicateCouplingId) {
      ++local_active_duplicate_reduced_corrections;
    } else if (reduced_update_has_distinct_left_right(update)) {
      local_has_distinct_multi_reduced_corrections = true;
    }
  }

  profile.active_face_corrections = local_active_face_corrections;
  profile.active_reduced_corrections = local_active_reduced_corrections;
  profile.active_duplicate_reduced_corrections = local_active_duplicate_reduced_corrections;
  if (lhs.commu.nTasks > 1) {
    int global_active_face_corrections = profile.active_face_corrections;
    int global_active_reduced_corrections = profile.active_reduced_corrections;
    int global_active_duplicate_reduced_corrections =
        profile.active_duplicate_reduced_corrections;
    fsils_allreduce_sum(&profile.active_face_corrections,
                        &global_active_face_corrections,
                        1,
                        cm_mod::mpint,
                        const_cast<FSILS_commuType&>(lhs.commu));
    fsils_allreduce_sum(&profile.active_reduced_corrections,
                        &global_active_reduced_corrections,
                        1,
                        cm_mod::mpint,
                        const_cast<FSILS_commuType&>(lhs.commu));
    fsils_allreduce_sum(&profile.active_duplicate_reduced_corrections,
                        &global_active_duplicate_reduced_corrections,
                        1,
                        cm_mod::mpint,
                        const_cast<FSILS_commuType&>(lhs.commu));
    profile.active_face_corrections = global_active_face_corrections;
    profile.active_reduced_corrections = global_active_reduced_corrections;
    profile.active_duplicate_reduced_corrections =
        global_active_duplicate_reduced_corrections;

    int local_distinct = local_has_distinct_multi_reduced_corrections ? 1 : 0;
    int global_distinct = local_distinct;
    fsils_allreduce_sum(&local_distinct,
                        &global_distinct,
                        1,
                        cm_mod::mpint,
                        const_cast<FSILS_commuType&>(lhs.commu));
    profile.has_distinct_multi_reduced_corrections = global_distinct != 0;
  } else {
    profile.has_distinct_multi_reduced_corrections = local_has_distinct_multi_reduced_corrections;
  }

  profile.active_nonduplicate_reduced_corrections =
      std::max(0, profile.active_reduced_corrections -
                      profile.active_duplicate_reduced_corrections);
  profile.reduced_touches_constraint =
      reduced_updates_touch_constraint_block(lhs, con_start, con_ncomp);
  profile.grouped_touches_constraint =
      grouped_bordered_touch_constraint_block(lhs, con_start, con_ncomp);
  return profile;
}

DistributedLowRankCorrection build(const FSILS_lhsType& lhs,
                                   int mom_start,
                                   int mom_ncomp,
                                   int con_start,
                                   int con_ncomp)
{
  DistributedLowRankCorrection correction;
  correction.profile = inspect(lhs, mom_start, mom_ncomp, con_start, con_ncomp);

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

    ProjectedReducedCorrection projected;
    projected.sigma = update.sigma;
    const auto& left_entries = projected_left_entries(update);
    const auto& right_entries = projected_right_entries(update);

    fill_projected_block_vector(lhs, left_entries, mom_start, mom_ncomp, projected.left_momentum);
    fill_projected_block_vector(lhs, left_entries, con_start, con_ncomp, projected.left_constraint);
    fill_projected_block_vector(lhs, right_entries, mom_start, mom_ncomp, projected.right_momentum);
    fill_projected_block_vector(lhs, right_entries, con_start, con_ncomp, projected.right_constraint);
    correction.reduced.push_back(std::move(projected));
  }

  for (const auto& group_data : lhs.grouped_bordered_field_couplings) {
    if (!group_data.active || group_data.modes.empty()) {
      continue;
    }

    const int rank = static_cast<int>(group_data.modes.size());
    if (group_data.aux_matrix.size() != static_cast<std::size_t>(rank * rank)) {
      continue;
    }

    ProjectedGroupedCorrection projected;
    projected.dense_coeff.assign(group_data.aux_matrix.size(), 0.0);
    if (!invert_dense_matrix(rank, group_data.aux_matrix, projected.dense_coeff)) {
      continue;
    }
    for (double& value : projected.dense_coeff) {
      value = -value;
    }

    projected.left_momentum_modes.reserve(group_data.modes.size());
    projected.left_constraint_modes.reserve(group_data.modes.size());
    projected.right_momentum_modes.reserve(group_data.modes.size());
    projected.right_constraint_modes.reserve(group_data.modes.size());

    for (const auto& mode : group_data.modes) {
      const auto& left_entries = projected_left_entries(mode);
      const auto& right_entries = projected_right_entries(mode);

      projected.left_momentum_modes.emplace_back();
      projected.left_constraint_modes.emplace_back();
      projected.right_momentum_modes.emplace_back();
      projected.right_constraint_modes.emplace_back();

      fill_projected_block_vector(lhs,
                                  left_entries,
                                  mom_start,
                                  mom_ncomp,
                                  projected.left_momentum_modes.back());
      fill_projected_block_vector(lhs,
                                  left_entries,
                                  con_start,
                                  con_ncomp,
                                  projected.left_constraint_modes.back());
      fill_projected_block_vector(lhs,
                                  right_entries,
                                  mom_start,
                                  mom_ncomp,
                                  projected.right_momentum_modes.back());
      fill_projected_block_vector(lhs,
                                  right_entries,
                                  con_start,
                                  con_ncomp,
                                  projected.right_constraint_modes.back());
    }
    correction.grouped.push_back(std::move(projected));
  }

  correction.profile.projected_mode_count = static_cast<int>(correction.reduced.size());
  for (const auto& group : correction.grouped) {
    correction.profile.projected_mode_count +=
        static_cast<int>(group.right_momentum_modes.size());
  }
  correction.profile.native_face_duplicates_only =
      !correction.profile.has_grouped_bordered &&
      correction.profile.active_face_corrections > 0 &&
      correction.profile.active_duplicate_reduced_corrections > 0 &&
      correction.profile.active_nonduplicate_reduced_corrections == 0 &&
      !correction.profile.has_explicit_corrections();
  return correction;
}

void apply_constraint_driven(FSILS_lhsType& lhs,
                             const DistributedLowRankCorrection& correction,
                             const Array<double>& in_constraint,
                             Array<double>& out_momentum,
                             Array<double>& out_constraint)
{
  const int n_reduced = static_cast<int>(correction.reduced.size());
  int total_rhs = n_reduced;
  for (const auto& group : correction.grouped) {
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
  for (const auto& corr : correction.reduced) {
    rhs[static_cast<std::size_t>(offset++)] =
        dense_dense_owned_dot_local(lhs, con_ncomp, corr.right_constraint, in_constraint);
  }
  for (const auto& group : correction.grouped) {
    for (const auto& mode : group.right_constraint_modes) {
      rhs[static_cast<std::size_t>(offset++)] =
          dense_dense_owned_dot_local(lhs, con_ncomp, mode, in_constraint);
    }
  }
  allreduce_sum_in_place(lhs, rhs);
  maybeTraceCorrectionCoefficients(lhs, "constraint_rhs", rhs, nullptr);

  offset = 0;
  std::vector<double> alpha_trace;
  alpha_trace.reserve(static_cast<std::size_t>(n_reduced));
  for (const auto& corr : correction.reduced) {
    const double alpha = corr.sigma * rhs[static_cast<std::size_t>(offset++)];
    alpha_trace.push_back(alpha);
    dense_axpy(mom_ncomp, nNo, corr.left_momentum, alpha, out_momentum);
    dense_axpy(con_ncomp, nNo, corr.left_constraint, alpha, out_constraint);
  }
  if (!alpha_trace.empty()) {
    maybeTraceCorrectionCoefficients(lhs, "constraint_reduced_alpha", rhs, &alpha_trace);
  }

  for (const auto& group : correction.grouped) {
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
      dense_axpy(mom_ncomp,
                 nNo,
                 group.left_momentum_modes[static_cast<std::size_t>(i)],
                 coeff,
                 out_momentum);
      dense_axpy(con_ncomp,
                 nNo,
                 group.left_constraint_modes[static_cast<std::size_t>(i)],
                 coeff,
                 out_constraint);
    }
  }
}

void apply_momentum_driven(FSILS_lhsType& lhs,
                           const DistributedLowRankCorrection& correction,
                           const Array<double>& in_momentum,
                           Array<double>& out_constraint)
{
  const int n_reduced = static_cast<int>(correction.reduced.size());
  int total_rhs = n_reduced;
  for (const auto& group : correction.grouped) {
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
  for (const auto& corr : correction.reduced) {
    rhs[static_cast<std::size_t>(offset++)] =
        dense_dense_owned_dot_local(lhs, mom_ncomp, corr.right_momentum, in_momentum);
  }
  for (const auto& group : correction.grouped) {
    for (const auto& mode : group.right_momentum_modes) {
      rhs[static_cast<std::size_t>(offset++)] =
          dense_dense_owned_dot_local(lhs, mom_ncomp, mode, in_momentum);
    }
  }
  allreduce_sum_in_place(lhs, rhs);
  maybeTraceCorrectionCoefficients(lhs, "momentum_rhs", rhs, nullptr);

  offset = 0;
  std::vector<double> alpha_trace;
  alpha_trace.reserve(static_cast<std::size_t>(n_reduced));
  for (const auto& corr : correction.reduced) {
    const double alpha = corr.sigma * rhs[static_cast<std::size_t>(offset++)];
    alpha_trace.push_back(alpha);
    dense_axpy(con_ncomp, nNo, corr.left_constraint, alpha, out_constraint);
  }
  if (!alpha_trace.empty()) {
    maybeTraceCorrectionCoefficients(lhs, "momentum_reduced_alpha", rhs, &alpha_trace);
  }

  for (const auto& group : correction.grouped) {
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
      dense_axpy(con_ncomp,
                 nNo,
                 group.left_constraint_modes[static_cast<std::size_t>(i)],
                 alpha[static_cast<std::size_t>(i)],
                 out_constraint);
    }
  }
}

}  // namespace fe_fsi_linear_solver::distributed_low_rank_correction
