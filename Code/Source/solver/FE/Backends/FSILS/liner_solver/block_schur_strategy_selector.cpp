#include "block_schur_strategy_selector.h"

#include <cstdlib>

namespace fe_fsi_linear_solver {

namespace {

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

}  // namespace

BlockSchurStrategySelection BlockSchurStrategySelector::select(
    const FSILS_lhsType& lhs,
    const distributed_low_rank_correction::Profile& low_rank_profile,
    int con_ncomp)
{
  BlockSchurStrategySelection selection;
  selection.native_face_duplicates_only = low_rank_profile.native_face_duplicates_only;

  const bool force_schur_bicgstab =
      env_enabled("SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_BICGSTAB");
  const bool force_schur_gmres =
      env_enabled("SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_GMRES") &&
      !force_schur_bicgstab;
  const bool force_exact_schur_path = force_schur_gmres;

  const bool disable_face_only_legacy =
      env_enabled("SVMP_FSILS_BLOCKSCHUR_DISABLE_FACE_ONLY_LEGACY") ||
      force_exact_schur_path;
  selection.use_face_only_legacy_scalar_schur =
      (con_ncomp == 1) &&
      (low_rank_profile.active_face_corrections > 0) &&
      (low_rank_profile.active_reduced_corrections == 0) &&
      !low_rank_profile.has_grouped_bordered &&
      !disable_face_only_legacy;

  selection.require_exact_momentum_low_rank_path =
      low_rank_profile.has_grouped_bordered ||
      (low_rank_profile.active_nonduplicate_reduced_corrections > 1 &&
       low_rank_profile.has_distinct_multi_reduced_corrections);

  selection.use_momentum_only_low_rank_legacy_scalar_schur =
      (con_ncomp == 1) &&
      (low_rank_profile.active_face_corrections == 0) &&
      (low_rank_profile.active_reduced_corrections > 0 ||
       low_rank_profile.has_grouped_bordered) &&
      !low_rank_profile.reduced_touches_constraint &&
      !low_rank_profile.grouped_touches_constraint &&
      !selection.require_exact_momentum_low_rank_path &&
      !force_exact_schur_path;

  selection.prefer_schur_gmres =
      low_rank_profile.has_grouped_bordered ||
      ((!selection.native_face_duplicates_only) &&
       low_rank_profile.active_face_corrections > 0) ||
      low_rank_profile.active_nonduplicate_reduced_corrections > 1;

  if (force_schur_gmres) {
    selection.prefer_schur_gmres = true;
  }
  if (force_schur_bicgstab) {
    selection.prefer_schur_gmres = false;
  }

  const bool disable_auto_coupled_outer_fgmres =
      env_enabled("SVMP_FSILS_DISABLE_COUPLED_OUTER_FGMRES");
  selection.auto_enable_coupled_outer_fgmres_scalar =
      low_rank_profile.distributed &&
      !selection.native_face_duplicates_only &&
      (low_rank_profile.projected_mode_count > 1 ||
       low_rank_profile.active_face_corrections > 1) &&
      env_enabled("SVMP_FSILS_AUTO_COUPLED_OUTER_FGMRES") &&
      !disable_auto_coupled_outer_fgmres;

  selection.force_enable_coupled_outer_fgmres_scalar =
      env_enabled("SVMP_FSILS_ENABLE_COUPLED_OUTER_FGMRES");
  selection.use_coupled_outer_fgmres_scalar =
      selection.auto_enable_coupled_outer_fgmres_scalar ||
      (selection.force_enable_coupled_outer_fgmres_scalar &&
       (low_rank_profile.active_face_corrections > 0 ||
        low_rank_profile.has_explicit_corrections()));

  return selection;
}

}  // namespace fe_fsi_linear_solver
