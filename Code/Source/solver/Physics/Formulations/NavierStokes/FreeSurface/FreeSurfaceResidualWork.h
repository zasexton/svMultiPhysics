/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_RESIDUAL_WORK_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_RESIDUAL_WORK_H

/**
 * @file FreeSurfaceResidualWork.h
 * @brief Residual-work channels of the free-surface energy ledger
 *
 * Each channel is an exact additive group of the installed residual,
 * installed as its own operator and declared to the FESystem; pairing its
 * constrained residual with the converged state times -dt gives the signed
 * work of that group in one accepted step.  Moved from
 * IncompressibleNavierStokesVMSModule::registerOn.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesVMSFormulation.h"

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

/// The forms of the residual-work channels: exact additive groups of the
/// installed residual.
struct FreeSurfaceResidualWorkForms {
    FE::forms::FormExpr convection;                 // moving control volume + convection
    FE::forms::FormExpr pressure;                   // -p div v
    FE::forms::FormExpr continuity;                 // q div u
    FE::forms::FormExpr nonconservative_body_force; // forcing minus its constant part
    FE::forms::FormExpr weak_boundary_momentum;     // Nitsche and kinematic penalty rows
    FE::forms::FormExpr weak_boundary_continuity;
    FE::forms::FormExpr vms_momentum;               // VMS minus Galerkin
    FE::forms::FormExpr vms_continuity;
    FE::forms::FormExpr vms_pspg_pressure_gradient;
    FE::forms::FormExpr ghost_penalty;              // cut-facet pressure penalty
};

struct FreeSurfaceResidualWorkApplicability {
    bool include_mcv{false};
    bool has_nonconservative_body_force{false};
    bool enable_vms{true};
};

/// Installs every produced channel as its own operator (operator tag + the
/// suffix of FreeSurfaceResidualWorkOperatorSuffixes) with copies of
/// @p install, and declares produced and inapplicable channels.
void installFreeSurfaceResidualWork(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const FreeSurfaceResidualWorkForms& work,
    const FreeSurfaceResidualWorkApplicability& applicability);

/// The bulk channels from the integrands and the volume forms M, C on @p liquid
/// (before any boundary term is added); @p conservative_body_force_residual is
/// the constant-gravity part -rho g.v of the forcing.
[[nodiscard]] FreeSurfaceResidualWorkForms bulkResidualWork(
    const NavierStokesVMSIntegrands& integrands,
    const FE::forms::FormExpr& momentum_form,
    const FE::forms::FormExpr& continuity_form,
    const FE::forms::FormExpr& vms_pspg_pressure_gradient_form,
    const FE::forms::FormExpr& conservative_body_force_residual,
    bool has_nonconservative_body_force,
    bool enable_vms,
    const FE::forms::VolumeDomain& liquid);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_RESIDUAL_WORK_H
