/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_LEGACY_TERMS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_LEGACY_TERMS_H

/**
 * @file FreeSurfaceLegacyTerms.h
 * @brief Free-surface terms kept for legacy configurations
 *
 * The unfitted CurvatureTraction with the level-set gradient normal (the
 * contact-line tests use it), and the fluid kinematic rows of schema-1
 * legacy enforcement on fitted boundaries (the penalty on (u - w).n and the
 * fitted Nitsche row), moved from IncompressibleNavierStokesVMSModule.cpp.
 * The production terms (SurfaceStress or the kinematic area-gradient
 * traction, MeshNitsche kinematics on the mesh rows) are stated in that file.
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

/// CurvatureTraction on an unfitted interface: -1 * ((-p_ext - gamma kappa) n) . v,
/// kappa = unfittedTractionCurvature (0 for a literal gamma = 0).
[[nodiscard]] FE::forms::FormExpr unfittedCurvatureTraction(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi,
    const FE::forms::FormExpr& p_ext,
    const FE::forms::FormExpr& gamma,
    const FE::forms::FormExpr& n,
    const FE::forms::FormExpr& v);

/// The schema-1 fluid kinematic row of a fitted free surface: the penalty on
/// (u - w).n, or the fitted Nitsche row; both also enter the weak-boundary
/// work channels.  MeshNitsche and None add no fluid row.
void addFittedFluidKinematicRow(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mesh_velocity,
    const FE::forms::FormExpr& mu,
    const FE::forms::FormExpr& n,
    bool ale_enabled,
    FE::forms::FormExpr* weak_boundary_momentum_work_form,
    FE::forms::FormExpr* weak_boundary_continuity_work_form);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_LEGACY_TERMS_H
