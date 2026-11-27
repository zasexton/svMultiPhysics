/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#pragma once

namespace svmp::FE::systems { class FESystem; }

namespace svmp::FE::level_set {

/// Labels of the generated geometry in logs and constraint manifests. Supply
/// string literals, or other strings that live as long as the reporting host.
struct LevelSetRecordLabels {
    const char* snapshot_heading{"authoritative level-set geometry"};
    const char* snapshot_diagnostic{"level_set_geometry_snapshot"};
    const char* nonempty_active_cells{"nonempty_active_cells"};
    const char* full_active_cells{"full_active_cells"};
    const char* inactive_cells{"inactive_cells"};
    const char* active_side{"active_side"};
    const char* inactive_vertices{"inactive_vertices"};
    const char* inactive_trace_support_vertices{"inactive_trace_support_vertices"};
    const char* max_inactive_extended_speed{"max_inactive_extended_speed"};
    const char* active_to_inactive_amplification{"active_to_inactive_amplification"};
    const char* active_to_inactive_amplification_guard{"active_to_inactive_amplification_guard"};
    const char* snapshot_revision{"level_set_snapshot_revision"};
    const char* active_volume{"active_volume"};
    const char* cut_cell_active_volume{"cut_cell_active_volume"};
    const char* full_cell_active_volume{"full_cell_active_volume"};
    const char* snapshot{"level-set snapshot"};
    const char* snapshot_validation{"level_set_snapshot_validation"};
    const char* aggregation_rootless_hint{
        "the candidate's cut feature has full cells only beyond the aggregation guards; "
        "SmallCutAggregationGuardOptions::rootless_fallback permits a homogeneous pin for "
        "each such candidate; every case is logged and counted in the run summary"};
};

inline constexpr LevelSetRecordLabels defaultLevelSetRecordLabels{};

/// Formulation-specific labels, when required by a published convergence record.
/// Strings must remain valid for the duration of the minimization.
struct LevelSetMinimizerRecordLabels {
    const char* method{"volume-constrained minimization"};
    const char* equilibrium{"volume_constrained_equilibrium"};
    const char* merit{"volume_merit"};
    const char* converged{"fixed_volume_equilibrium_converged"};
};

/// A formulation or application may attach its reporting labels to the system.
[[nodiscard]] const LevelSetRecordLabels& levelSetRecordLabels(const systems::FESystem& system) noexcept;

} // namespace svmp::FE::level_set
