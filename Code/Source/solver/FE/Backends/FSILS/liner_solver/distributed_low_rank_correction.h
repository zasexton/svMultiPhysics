#ifndef SV_FE_LS_DISTRIBUTED_LOW_RANK_CORRECTION_H
#define SV_FE_LS_DISTRIBUTED_LOW_RANK_CORRECTION_H

#include "fils_struct.hpp"

namespace fe_fsi_linear_solver::distributed_low_rank_correction {

struct ProjectedReducedCorrection {
  double sigma{0.0};
  Array<double> left_momentum;
  Array<double> left_constraint;
  Array<double> right_momentum;
  Array<double> right_constraint;
};

struct ProjectedGroupedCorrection {
  std::vector<double> dense_coeff;
  std::vector<Array<double>> left_momentum_modes;
  std::vector<Array<double>> left_constraint_modes;
  std::vector<Array<double>> right_momentum_modes;
  std::vector<Array<double>> right_constraint_modes;
};

struct Profile {
  bool distributed{false};
  bool has_grouped_bordered{false};
  bool reduced_touches_constraint{false};
  bool grouped_touches_constraint{false};
  bool has_distinct_multi_reduced_corrections{false};
  int active_face_corrections{0};
  int active_reduced_corrections{0};
  int active_duplicate_reduced_corrections{0};
  int active_nonduplicate_reduced_corrections{0};
  int projected_mode_count{0};
  bool native_face_duplicates_only{false};

  [[nodiscard]] bool has_explicit_corrections() const noexcept
  {
    return projected_mode_count > 0;
  }
};

struct DistributedLowRankCorrection {
  Profile profile{};
  std::vector<ProjectedReducedCorrection> reduced;
  std::vector<ProjectedGroupedCorrection> grouped;

  [[nodiscard]] bool empty() const noexcept
  {
    return reduced.empty() && grouped.empty();
  }
};

Profile inspect(const FSILS_lhsType& lhs,
                int mom_start,
                int mom_ncomp,
                int con_start,
                int con_ncomp);

DistributedLowRankCorrection build(const FSILS_lhsType& lhs,
                                   int mom_start,
                                   int mom_ncomp,
                                   int con_start,
                                   int con_ncomp);

void apply_constraint_driven(FSILS_lhsType& lhs,
                             const DistributedLowRankCorrection& correction,
                             const Array<double>& in_constraint,
                             Array<double>& out_momentum,
                             Array<double>& out_constraint);

void apply_momentum_driven(FSILS_lhsType& lhs,
                           const DistributedLowRankCorrection& correction,
                           const Array<double>& in_momentum,
                           Array<double>& out_constraint);

void trace_momentum_projection(FSILS_lhsType& lhs,
                               const DistributedLowRankCorrection& correction,
                               const char* label,
                               const Array<double>& in_momentum,
                               bool use_left = false);

}  // namespace fe_fsi_linear_solver::distributed_low_rank_correction

#endif  // SV_FE_LS_DISTRIBUTED_LOW_RANK_CORRECTION_H
