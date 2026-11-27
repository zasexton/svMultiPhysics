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
    .active_side = "liquid_side",
    .inactive_vertices = "dry_vertices",
    .inactive_trace_support_vertices = "dry_trace_support_vertices",
    .max_inactive_extended_speed = "max_dry_extended_speed",
    .active_to_inactive_amplification = "wet_to_dry_amplification",
    .active_to_inactive_amplification_guard = "wet_to_dry_amplification_guard",
    .snapshot_revision = "free_surface_snapshot_revision",
    .active_volume = "active_wet_volume",
    .cut_cell_active_volume = "cut_cell_active_wet_volume",
    .full_cell_active_volume = "full_cell_active_wet_volume"};

inline constexpr FE::level_set::ConservativePhaseRecordSchema conservativePhaseRecordSchema{
    .version = 3,
    .active_side = "liquid_side",
    .measure = "liquid_measure",
    .indicator = "liquid_indicator",
    .nonprimary_component = "resolved_satellite"};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
