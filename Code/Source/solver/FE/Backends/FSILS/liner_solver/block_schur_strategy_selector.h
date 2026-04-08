#ifndef SV_FE_LS_BLOCK_SCHUR_STRATEGY_SELECTOR_H
#define SV_FE_LS_BLOCK_SCHUR_STRATEGY_SELECTOR_H

#include "distributed_low_rank_correction.h"

namespace fe_fsi_linear_solver {

struct BlockSchurStrategySelection {
  bool use_face_only_legacy_scalar_schur{false};
  bool use_momentum_only_low_rank_legacy_scalar_schur{false};
  bool require_exact_momentum_low_rank_path{false};
  bool prefer_schur_gmres{false};
  bool native_face_duplicates_only{false};
  bool auto_enable_coupled_outer_fgmres_scalar{false};
  bool force_enable_coupled_outer_fgmres_scalar{false};
  bool use_coupled_outer_fgmres_scalar{false};

  [[nodiscard]] bool use_legacy_scalar_schur() const noexcept
  {
    return use_face_only_legacy_scalar_schur ||
           use_momentum_only_low_rank_legacy_scalar_schur;
  }
};

class BlockSchurStrategySelector {
 public:
  static BlockSchurStrategySelection select(
      const FSILS_lhsType& lhs,
      const distributed_low_rank_correction::Profile& low_rank_profile,
      int con_ncomp);
};

}  // namespace fe_fsi_linear_solver

#endif  // SV_FE_LS_BLOCK_SCHUR_STRATEGY_SELECTOR_H
