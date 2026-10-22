/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_FITTED_A_L_E_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_FITTED_A_L_E_H

/**
 * @file FreeSurfaceFittedALE.h
 * @brief Declarations a fitted ALE free surface makes to the mesh-motion system
 *
 * Moved from IncompressibleNavierStokesVMSModule.cpp: the strong pin of a
 * fitted contact line, and the normal kinematic relations that the
 * mesh-motion module turns into its Nitsche consistency term.
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

void applyFreeSurfaceContactLineConstraints(
    FE::constraints::ConstraintPlan& constraint_plan,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    int dim);

/// The normal kinematic relations (dt d - u).n = 0 of the fitted boundaries,
/// collected before the residual is installed.
[[nodiscard]] std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
fittedNormalKinematicDeclarations(
    const FE::systems::ALEBinding& ale_binding,
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces,
    const FE::forms::FormExpr& u,
    FE::FieldId u_id);

/// Declares them after the residual is installed, with their consumers and
/// operator-stage measurements.
void declareFittedNormalKinematics(
    FE::systems::FESystem& system,
    const std::string& operator_tag,
    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
        normal_declarations,
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_FITTED_A_L_E_H
