/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_DIAGNOSTICS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_DIAGNOSTICS_H

/**
 * @file FreeSurfaceDiagnostics.h
 * @brief Log lines of the free-surface terms (FE_LOG_INFO / FE_LOG_WARNING)
 *
 * The log statements of the free-surface terms, moved from
 * IncompressibleNavierStokesVMSModule.cpp with their text unchanged: scripts
 * and tests parse the diagnostic=... tokens.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FormsInstaller.h"

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void logUnfittedFreeSurfaceMode(const FreeSurfaceBoundary& bc);

/// Natural mode: no interface term (sigma n = 0).
void logNaturalUnfittedFreeSurface(const FreeSurfaceBoundary& bc);

/// The installed traction (SurfaceStress, KinematicAreaGradientTraction) and
/// the semi-implicit capillary term.
void logFreeSurfaceTraction(const FreeSurfaceBoundary& bc);

/// The terms of one contact line (@p active_wall_marker: the wetted wall, or -1).
void logContactLineTerms(
    const FreeSurfaceBoundary& bc,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker,
    int active_wall_marker);

void logCutCellStabilization(
    const FreeSurfaceBoundary& bc,
    int velocity_polynomial_order,
    int pressure_polynomial_order);

/// Registration order of the inactive-side and aggregation constraints
/// (@p liquid_boundary null: none are registered).
void logActiveDomainConstraints(
    const FreeSurfaceBoundary* liquid_boundary,
    bool small_cut_aggregation);

void logStrongContactPointLaw(const FreeSurfaceContactLine& contact_line);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_DIAGNOSTICS_H
