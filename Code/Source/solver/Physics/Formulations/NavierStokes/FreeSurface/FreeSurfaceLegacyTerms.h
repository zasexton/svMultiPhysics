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
 * contact-line tests use it), moved from IncompressibleNavierStokesVMSModule.cpp.
 * The fluid kinematic rows of the schema-1 Penalty and Nitsche enforcements
 * on fitted boundaries were retired by decision D47 Q9.  The production
 * terms (SurfaceStress or the kinematic area-gradient traction, MeshNitsche
 * kinematics on the mesh rows) are stated in that file.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Forms/FormExpr.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

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

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_LEGACY_TERMS_H
