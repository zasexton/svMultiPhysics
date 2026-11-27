/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#pragma once

#include "FE/LevelSet/ConservativePhaseRecordSchema.h"
#include "FE/LevelSet/LevelSetRecordLabels.h"

namespace svmp::Physics::formulations::navier_stokes::free_surface {

inline constexpr FE::level_set::LevelSetRecordLabels levelSetRecordLabels{
    .snapshot_heading = "authoritative free-surface geometry",
    .snapshot_diagnostic = "free_surface_geometry_snapshot",
    .nonempty_active_cells = "active_wet_cells",
    .full_active_cells = "active_full_wet_cells",
    .inactive_cells = "active_full_dry_cells",
    .active_side = "liquid_side"};

inline constexpr FE::level_set::ConservativePhaseRecordSchema conservativePhaseRecordSchema{
    .version = 3,
    .active_side = "liquid_side",
    .measure = "liquid_measure",
    .indicator = "liquid_indicator"};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
