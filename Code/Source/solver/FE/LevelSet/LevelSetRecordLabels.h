/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#pragma once

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
};

inline constexpr LevelSetRecordLabels defaultLevelSetRecordLabels{};

} // namespace svmp::FE::level_set
