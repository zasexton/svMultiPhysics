/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_PREFLIGHT_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_PREFLIGHT_H

/**
 * @file NavierStokesPreflight.h
 * @brief Checks of a Navier-Stokes registration before the system is changed
 *
 * Moved from the beginning of IncompressibleNavierStokesVMSModule::registerOn.
 * The choice of an incremental PSPG pressure-gradient form went with
 * KinematicAreaGradientTraction, its only selector (decision D48).
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

#include "FE/Spaces/FunctionSpace.h"

#include <memory>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// What preflightNavierStokes resolved for the registration that follows.
struct NavierStokesPreflight {
    int dim{0};
    bool enable_vms{true};
    FE::systems::FieldSpec u_spec;
    FE::systems::FieldSpec p_spec;
    std::optional<FE::systems::FieldSpec> body_force_spec;
    std::optional<FE::systems::FieldSpec> semi_implicit_reference_spec;
    FE::systems::ALEBindingOptions ale_options;
    std::vector<FreeSurfaceBoundary> effective_free_surfaces;   // generated interface markers resolved
    bool has_exterior_free_surface{false};
};

/// Every check of a Navier-Stokes registration that runs before the system is
/// changed: options, spaces, free surfaces (capability rules, fitted
/// mesh-motion ownership, generated markers, contact walls, energy
/// declarations), generated-boundary Nitsche routes, field compatibility.  A
/// rejected configuration leaves the system definition unchanged.
[[nodiscard]] NavierStokesPreflight preflightNavierStokes(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& pressure_space);

/// The system has a native mesh, which the active-side constraints need;
/// warns (and the constraints are skipped) otherwise.
[[nodiscard]] bool hasNativeMeshOrWarn(const FE::systems::FESystem& system);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_PREFLIGHT_H
