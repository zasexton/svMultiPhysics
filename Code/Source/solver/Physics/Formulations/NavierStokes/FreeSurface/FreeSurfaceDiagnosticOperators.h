/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_DIAGNOSTIC_OPERATORS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_DIAGNOSTIC_OPERATORS_H

/**
 * @file FreeSurfaceDiagnosticOperators.h
 * @brief Opt-in diagnostic operators of the Navier-Stokes residual
 *
 * Installed after the production operators, only when requested through the
 * existing environment switches: SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC
 * (the conservative split used by the static-capillary path) and
 * SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC (the continuity-row parts used
 * by the cut-stability tests).  Moved from IncompressibleNavierStokesVMSModule.cpp.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"

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

bool pressureRowContributionDiagnosticEnabled();

bool freeSurfaceConservativeBalanceDiagnosticEnabled();

/// The conservative-balance diagnostic is requested and every free surface
/// has a discrete-energy traction with a constant nonzero gamma (warns when
/// it is requested but not supported).
[[nodiscard]] bool conservativeBalanceDiagnosticSupported(
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces,
    bool has_exterior_free_surface);

/// The operators of the conservative split
/// (FreeSurfaceConservativeBalanceDiagnosticOperators), on the production rules.
void installConservativeBalanceDiagnosticOperators(
    FE::systems::FESystem& system,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const FE::forms::FormExpr& free_surface_conservative_pressure_form,
    const FE::forms::FormExpr& free_surface_conservative_external_pressure_form,
    const FE::forms::FormExpr& free_surface_conservative_surface_energy_form,
    const FE::forms::FormExpr&
        free_surface_conservative_gravitational_potential_form,
    const FE::forms::FormExpr& pressure,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& q,
    const std::optional<ActiveVolumeDomain>& active_volume_domain);

/// The continuity-row parts used by the cut-stability tests
/// (equations_diagnostic_ns_*).
void installPressureRowContributionDiagnosticOperators(
    FE::systems::FESystem& system,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const FE::forms::FormExpr& galerkin_continuity_integrand,
    const FE::forms::FormExpr& active_continuity_diagnostic_form,
    const FE::forms::FormExpr& vms_pspg_continuity_integrand,
    const FE::forms::FormExpr& vms_pspg_pressure_gradient_integrand,
    const FE::forms::FormExpr& vms_pspg_nonpressure_integrand,
    const FE::forms::FormExpr& pressure_ghost_penalty_form,
    const std::optional<ActiveVolumeDomain>& active_volume_domain);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_DIAGNOSTIC_OPERATORS_H
