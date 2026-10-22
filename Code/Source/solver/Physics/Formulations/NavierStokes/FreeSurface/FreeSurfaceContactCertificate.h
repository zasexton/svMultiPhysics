/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_CONTACT_CERTIFICATE_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_CONTACT_CERTIFICATE_H

/**
 * @file FreeSurfaceContactCertificate.h
 * @brief Geometric certificate of the walls that carry unfitted contact lines
 *
 * A contact-line law needs a planar wall whose configured normal n_w matches
 * the mapped wall facets and an interface transverse to it.  Moved from
 * IncompressibleNavierStokesVMSModule.cpp: the planar-wall evidence (with
 * collective reductions), the mapped contact covectors and normals, the
 * preflight of every contact wall, the axis-aligned normal-only wall
 * condition, and the refresh-time checks registered as cut-integration-context
 * callbacks for prescribed-angle and dynamic contact lines.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Core/Types.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <span>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

constexpr FE::Real kMinimumContactTransverseSine = FE::Real{1.0e-6};

void validateDynamicContactPlanarWallMarker(
    const FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line);

void validateFreeSurfaceContactGeometryPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system);

void validateDynamicContactWallEssentialBC(
    const IncompressibleNavierStokesVMSOptions& options,
    const FreeSurfaceContactLine& contact_line,
    int dim);

void registerFreeSurfacePrescribedAngleGeometry(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc);

/// Registers the refresh-time wall-normal check of a dynamic contact line and
/// checks the current cut context once.
void registerDynamicContactWallNormalCheck(
    FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_CONTACT_CERTIFICATE_H
