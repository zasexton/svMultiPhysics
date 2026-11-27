/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#pragma once

#include "FE/LevelSet/ConservativePhaseRecordSchema.h"
#include "FE/LevelSet/LevelSetRecordLabels.h"

namespace svmp::Physics::formulations::navier_stokes::free_surface {

inline constexpr char activeBoundarySubsetLabel[] = "wetted";

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
    .full_cell_active_volume = "full_cell_active_wet_volume",
    .support_refresh_heading = "active pressure support constraint refresh",
    .support_refresh_diagnostic = "active_pressure_constraint_refresh",
    .extension_factorization_source = "DryRegionMumps",
    .snapshot = "free-surface snapshot",
    .snapshot_validation = "free_surface_snapshot_validation",
    .aggregation_rootless_hint =
        "setting "
        "<Small_cut_aggregation_rootless_fallback>true</Small_cut_aggregation_rootless_fallback> "
        "on the free-surface boundary condition enables the rootless-island policy. "
        "It pins velocity and pressure at the affected candidates, which degrades the "
        "solution in connected liquid. The run summary records accepted pins"};

inline constexpr FE::level_set::LevelSetMinimizerRecordLabels staticCapillaryMinimizerRecords{
    .method = "static capillary equilibrium minimization",
    .equilibrium = "static_capillary_equilibrium",
    .merit = "capillary_merit",
    .converged = "fixed_volume_discrete_capillary_equilibrium_converged"};

inline constexpr FE::level_set::ConservativePhaseRecordSchema conservativePhaseRecordSchema{
    .version = 3,
    .active_side = "liquid_side",
    .measure = "liquid_measure",
    .indicator = "liquid_indicator",
    .nonprimary_component = "resolved_satellite"};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
