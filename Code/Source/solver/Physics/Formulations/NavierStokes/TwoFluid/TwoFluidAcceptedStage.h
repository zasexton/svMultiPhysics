/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

/**
 * @file
 * @brief Accepted-stage diagnostics of the incompressible two-fluid
 *        interface: phase diagnostics at the current stage, the momentum
 *        reconciliation of conservative-phase maintenance and the
 *        accepted-stage solve telemetry.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRunSupport.h"

#include "FE/Core/Types.h"
#include "FE/Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "FE/LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::two_fluid {

using free_surface::FreeSurfaceSystemHandle;

std::vector<svmp::FE::systems::AcceptedTwoFluidStageDiagnosticState>
evaluateCurrentTwoFluidStageDiagnostics(
    const FreeSurfaceSystemHandle& sim,
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
    const FreeSurfaceSystemHandle& sim,
    std::span<const svmp::FE::level_set::ConservativePhaseRequest>
        maintenance_requests,
    std::span<const svmp::FE::Real> corrected_solution,
    std::uint64_t corrected_algebraic_revision,
    svmp::FE::Real effective_dt,
    const svmp::MeshComm& comm);

svmp::FE::systems::TwoFluidAcceptedStageSolveTelemetry
makeTwoFluidAcceptedStageSolveTelemetry(
    const svmp::FE::timestepping::NewtonReport& report);

} // namespace svmp::Physics::formulations::navier_stokes::two_fluid
