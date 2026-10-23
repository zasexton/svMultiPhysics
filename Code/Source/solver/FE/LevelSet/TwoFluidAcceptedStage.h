/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_LEVELSET_TWO_FLUID_ACCEPTED_STAGE_H
#define SVMP_FE_LEVELSET_TWO_FLUID_ACCEPTED_STAGE_H

/**
 * @file
 * @brief Accepted-stage diagnostics of the incompressible two-fluid
 *        interface: phase diagnostics at the current stage, the momentum
 *        reconciliation of conservative-phase maintenance and the
 *        accepted-stage solve telemetry.
 */

#include "Core/Types.h"
#include "Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "Systems/FESystem.h"
#include "Systems/FESystemHandle.h"
#include "TimeStepping/NewtonSolver.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <span>
#include <vector>

namespace svmp::FE::level_set {

std::vector<svmp::FE::systems::AcceptedTwoFluidStageDiagnosticState>
evaluateCurrentTwoFluidStageDiagnostics(
    const svmp::FE::systems::FESystemHandle& sim,
    std::span<const svmp::FE::systems::
                        TwoFluidAcceptedStageDiagnosticDeclaration>
        declarations,
    std::span<const svmp::FE::Real> stage_solution,
    svmp::FE::Real effective_dt,
    const svmp::MeshComm& comm);

struct TwoFluidMomentumReconciliationBatch {
  std::vector<svmp::FE::interfaces::
                  IncompressibleTwoFluidMomentumReconciliation>
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

svmp::FE::systems::TwoFluidAcceptedStageSolveTelemetry
makeTwoFluidAcceptedStageSolveTelemetry(
    const svmp::FE::timestepping::NewtonReport& report);

} // namespace svmp::FE::level_set

#endif // SVMP_FE_LEVELSET_TWO_FLUID_ACCEPTED_STAGE_H
