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
 * @brief The accepted free-surface contact stages are FE services: the
 *        operator-stage capture (FE/TimeStepping/OperatorStageCapture.h),
 *        the contact-line stage observables
 *        (FE/Interfaces/ContactLineStageObservables.h) and the accepted
 *        wall-contact cells (FE/LevelSet/Maintenance/
 *        LevelSetWallAwareMaintenance.h).  These forwards keep the callers
 *        until they name the FE services; the two functions below pass the
 *        free-surface functional declarations of the records.
 */

#include "FE/Core/Types.h"
#include "FE/Interfaces/ContactLineStageObservables.h"
#include "FE/LevelSet/AcceptedFunctionalRecorder.h"
#include "FE/LevelSet/LevelSetContactStageParticipant.h"
#include "FE/LevelSet/Maintenance/LevelSetWallAwareMaintenance.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FESystemHandle.h"
#include "FE/Systems/FunctionalRecords.h"
#include "FE/TimeStepping/OperatorStageCapture.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using svmp::FE::interfaces::firstOrderGeneralizedAlphaContactProvenanceValid;
using svmp::FE::interfaces::contactStageRealValuesMatch;
using svmp::FE::timestepping::DynamicContactOperatorStageFieldSlice;
using svmp::FE::timestepping::DynamicContactOperatorStageState;
using svmp::FE::timestepping::DynamicContactFirstOrderGeneralizedAlphaObservation;
using svmp::FE::timestepping::dynamicContactStageMeshRevisionMatches;
using svmp::FE::timestepping::dynamicContactWallBoundaryFingerprint;
using svmp::FE::timestepping::dynamicContactStageProvenanceMatches;
using svmp::FE::timestepping::makeDynamicContactFirstOrderGeneralizedAlphaObservation;
using svmp::FE::timestepping::captureDynamicContactOperatorStageState;
using svmp::FE::timestepping::overlayDynamicContactOperatorStageState;
using svmp::FE::timestepping::requireDynamicContactOperatorStageObservationConsensus;
using svmp::FE::timestepping::FreeSurfaceContactStageTemporalCoordinates;
using svmp::FE::timestepping::resolveFreeSurfaceContactStageTemporalCoordinates;
using svmp::FE::interfaces::acceptedContactStageRevision;
using svmp::FE::interfaces::bindAcceptedFreeSurfaceContactStagesToEndpointRevision;
using svmp::FE::level_set::assertAcceptedFreeSurfaceContactStageEndpointUnchanged;

inline std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
evaluateAcceptedFreeSurfaceContactStages(
    const svmp::FE::systems::FESystemHandle& sim,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution,
    const std::optional<svmp::FE::systems::
                            FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt)
{
  return svmp::FE::interfaces::evaluateAcceptedFreeSurfaceContactStages(
      *sim.fe_system,
      svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations(),
      stage_time,
      stage_alpha_f,
      previous_state_revision,
      endpoint_state_revision,
      stage_solution,
      first_order_generalized_alpha);
}

inline std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
captureAcceptedContactStageWallConstraints(
    const svmp::FE::systems::FESystemHandle& sim,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages)
{
  return svmp::FE::level_set::captureAcceptedContactStageWallConstraints(
      *sim.fe_system,
      svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations(),
      accepted_contact_stages);
}

inline void recordInitialFreeSurfaceDiscreteFunctionalBaseline(
    const svmp::FE::systems::FESystemHandle& sim,
    std::uint64_t initial_step,
    svmp::FE::Real initial_time,
    std::span<const svmp::FE::Real> initial_solution,
    bool record_backward_euler_kinetic_baseline = false,
    std::optional<std::uint64_t>
        initial_extension_map_revision = std::nullopt)
{
  svmp::FE::level_set::recordInitialAcceptedFunctionalBaseline(
      sim,
      initial_step,
      initial_time,
      initial_solution,
      record_backward_euler_kinetic_baseline,
      initial_extension_map_revision);
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
