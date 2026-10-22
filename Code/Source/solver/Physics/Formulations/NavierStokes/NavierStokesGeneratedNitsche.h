/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_GENERATED_NITSCHE_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_GENERATED_NITSCHE_H

/**
 * @file NavierStokesGeneratedNitsche.h
 * @brief Certification bookkeeping of weak velocity conditions on generated wetted boundaries
 *
 * A symmetric Nitsche condition on the wetted part of a wall of a cut liquid
 * is admitted only with small-cut aggregation and an aggregate-trace
 * certificate: the form bindings returned by applyVelocityNitscheBCs are
 * counted against the routes here, and
 * FE::systems::FormulationBuilder::certifyGeneratedBoundaryNitscheTraces
 * passes them to the installer, which checks the minimum symmetric energy
 * ratio at every refresh.  Moved from IncompressibleNavierStokesVMSModule::registerOn.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FormsInstaller.h"

#include <optional>
#include <vector>
#include <functional>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// Every weak velocity condition on a generated wetted boundary produced one
/// certified form binding (only with small-cut aggregation and constant
/// Newtonian viscosity).
void requireCertifiedNitscheTraces(
    const IncompressibleNavierStokesVMSOptions& options,
    const std::function<std::optional<int>(int)>& generated_active_boundary_for,
    bool small_cut_aggregation,
    const std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>&
        generated_nitsche_trace_bindings);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_GENERATED_NITSCHE_H
