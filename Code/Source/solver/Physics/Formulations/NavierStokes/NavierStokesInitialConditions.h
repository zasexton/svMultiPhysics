/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_INITIAL_CONDITIONS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_INITIAL_CONDITIONS_H

/**
 * @file NavierStokesInitialConditions.h
 * @brief Initial velocity and pressure, and node pressure constraints on the liquid support
 *
 * Moved from IncompressibleNavierStokesVMSModule.cpp: the mesh vertex-field
 * initialization of the unknowns, the liquid support of the pressure (the
 * vertices of the active side, including cut cells), the hydrostatic
 * initialization on that support, and the preflight check that node
 * pressure constraints lie on it.
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

#include "FE/Backends/Interfaces/GenericVector.h"

#include <functional>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void validateActiveDomainPressureConstraints(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::vector<FreeSurfaceBoundary>& free_surfaces);

/// Initial velocity and pressure: mesh vertex fields named like the unknowns,
/// and, with hydrostatic initialization, @p hydrostatic_pressure on the liquid
/// support (the reference pressure on dry vertices) or a level-set-aware vertex
/// field; checks the node pressure constraints against it
/// (diagnostic=hydrostatic_initialization).
void initializeNavierStokesState(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>&
        hydrostatic_pressure);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_INITIAL_CONDITIONS_H
