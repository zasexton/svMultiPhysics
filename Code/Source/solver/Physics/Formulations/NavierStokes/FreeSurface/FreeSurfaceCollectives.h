/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_COLLECTIVES_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_COLLECTIVES_H

/**
 * @file FreeSurfaceCollectives.h
 * @brief Rank-independent agreement on the local phase of a collective callback
 *
 * A cut-integration-context callback that reduces over the communicator
 * first agrees, with MPI_MIN, whether every rank reached the reduction; a
 * rank-local failure is logged with its rank before the collective so that
 * every rank fails at the same point with the same message.  Moved from
 * IncompressibleNavierStokesVMSModule.cpp.
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
#include <exception>
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

[[maybe_unused]] void logRankLocalCollectiveFailure(
    std::string_view phase,
    const std::exception_ptr& failure);

[[nodiscard]] bool coordinateCutContextCallbackLocalPhase(
    const FE::systems::FESystem& system,
    const std::exception_ptr& local_exception,
    bool local_available,
    std::string_view phase);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_COLLECTIVES_H
