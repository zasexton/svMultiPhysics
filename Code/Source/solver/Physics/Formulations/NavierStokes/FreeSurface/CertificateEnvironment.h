/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#ifndef SVMP_PHYSICS_FREE_SURFACE_CERTIFICATE_ENVIRONMENT_H
#define SVMP_PHYSICS_FREE_SURFACE_CERTIFICATE_ENVIRONMENT_H

#include "FE/Analysis/ConstraintRangeCertificate.h"
#include "FE/Core/EnvironmentNames.h"

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// The solver catalogue extended by the free-surface formulation.
[[nodiscard]] const FE::environment::EnvironmentNameRegistry& environmentRegistry();
[[nodiscard]] bool keepFullCellPointsFromEnvironment();
[[nodiscard]] const std::vector<std::string>& rowContributionDiagnosticSwitches();
[[nodiscard]] const std::vector<std::string>& conservativeBalanceDiagnosticSwitches();
[[nodiscard]] FE::analysis::NewtonAnalysisMethod newtonAnalysisMethodFromEnvironment();

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

#endif // SVMP_PHYSICS_FREE_SURFACE_CERTIFICATE_ENVIRONMENT_H
