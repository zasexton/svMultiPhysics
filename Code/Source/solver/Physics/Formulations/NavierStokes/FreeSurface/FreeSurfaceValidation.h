/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_VALIDATION_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_VALIDATION_H

/**
 * @file FreeSurfaceValidation.h
 * @brief Capability rules of the free-surface and boundary options (fail closed)
 *
 * Moved from IncompressibleNavierStokesVMSModule.cpp: which combinations of
 * implementation, active domain, surface-tension form, kinematic
 * enforcement, contact laws, cut stabilization and aggregation are
 * supported, the standard boundary conditions against the free surfaces,
 * the uniqueness of generated markers, and the semi-implicit capillary
 * term's envelope, with the messages tests check.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void validateGeneratedFreeSurfaceMarkerUniqueness(
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system);

void validateFreeSurfaceBoundary(const FreeSurfaceBoundary& bc,
                                 const IncompressibleNavierStokesVMSOptions& options,
                                 bool ale_enabled,
                                 const FE::systems::FESystem& system,
                                 int dim);

void validateNavierStokesBoundaryConfiguration(
    const IncompressibleNavierStokesVMSOptions& options,
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    const FE::spaces::FunctionSpace& velocity_space,
    const FE::spaces::FunctionSpace& pressure_space,
    int dim);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_VALIDATION_H
