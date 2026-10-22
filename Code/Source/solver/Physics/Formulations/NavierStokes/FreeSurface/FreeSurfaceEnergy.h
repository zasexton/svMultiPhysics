/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_ENERGY_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_ENERGY_H

/**
 * @file FreeSurfaceEnergy.h
 * @brief Energy declarations of a free surface for the accepted-stage energy ledger
 *
 * Moved from IncompressibleNavierStokesVMSModule.cpp: the discrete surface,
 * wall (Young) and Ren-E functionals of the unfitted free surfaces and their
 * preflight, and the applicability of the external boundary energy.
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

void validateFreeSurfaceDiscreteFunctionalPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system);

void declareFreeSurfaceDiscreteFunctionals(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    FE::systems::FESystem& system,
    FE::FieldId velocity_field,
    FE::Real density,
    const std::array<FE::Real, 3>& gravitational_acceleration,
    bool static_conservative_body_force_complete,
    FE::Real dynamic_viscosity,
    bool has_constant_dynamic_viscosity);

/// Applicability of the external boundary energy of a free-surface system:
/// imposed traction, open-boundary flux, homogeneous velocity data, pressure
/// Dirichlet data.
void declareFreeSurfaceExternalBoundaryEnergy(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_ENERGY_H
