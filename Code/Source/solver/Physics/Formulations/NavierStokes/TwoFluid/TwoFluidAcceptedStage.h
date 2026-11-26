/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_PHYSICS_TWO_FLUID_TWO_FLUID_ACCEPTED_STAGE_H
#define SVMP_PHYSICS_TWO_FLUID_TWO_FLUID_ACCEPTED_STAGE_H

/**
 * @file
 * @brief Accepted-stage diagnostics of the incompressible two-fluid
 *        interface: phase diagnostics at the current stage, the momentum
 *        reconciliation of conservative-phase maintenance and the
 *        accepted-stage solve telemetry.
 */

#include "Physics/Formulations/NavierStokes/TwoFluid/TwoFluidRecordTypes.h"
#include "FE/Core/Types.h"
#include "Physics/Formulations/NavierStokes/TwoFluid/IncompressibleTwoFluidDiagnostics.h"
#include "FE/LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FESystemHandle.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::two_fluid {

std::vector<AcceptedTwoFluidStageDiagnosticState>
evaluateCurrentTwoFluidStageDiagnostics(
    const svmp::FE::systems::FESystemHandle& sim,
    std::span<const TwoFluidAcceptedStageDiagnosticDeclaration>
        declarations,
    std::span<const svmp::FE::Real> stage_solution,
    svmp::FE::Real effective_dt,
    const svmp::MeshComm& comm);

struct TwoFluidMomentumReconciliationBatch {
  std::vector<IncompressibleTwoFluidMomentumReconciliation>
      records{};
  bool satisfied{true};
};

TwoFluidMomentumReconciliationBatch
buildCurrentTwoFluidMomentumReconciliations(
    const svmp::FE::systems::FESystemHandle& sim,
    std::span<const svmp::FE::level_set::ConservativePhaseRequest>
        maintenance_requests,
    std::span<const svmp::FE::Real> corrected_solution,
    std::uint64_t corrected_algebraic_revision,
    svmp::FE::Real effective_dt,
    const svmp::MeshComm& comm);

TwoFluidAcceptedStageSolveTelemetry
makeTwoFluidAcceptedStageSolveTelemetry(
    const svmp::FE::timestepping::NewtonReport& report);

} // namespace svmp::Physics::formulations::navier_stokes::two_fluid

#endif // SVMP_PHYSICS_TWO_FLUID_TWO_FLUID_ACCEPTED_STAGE_H
