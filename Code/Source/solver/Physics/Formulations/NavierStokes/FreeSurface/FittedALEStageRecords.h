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
 * @brief The fitted-ALE normal operator-stage measurement records of an FE
 *        system, under the names of the fitted-ALE physics.
 *
 * The records are the FE mesh-normal constraint operator-stage measurements
 * (FE/Systems/MeshNormalConstraintStageMeasurements.h); fittedALERecords is
 * meshNormalConstraintStageMeasurements.
 */

#include "FE/Systems/FESystem.h"
#include "FE/Systems/MeshNormalConstraintStageMeasurements.h"

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using FittedALEStageRecordStore =
    svmp::FE::systems::MeshNormalConstraintStageMeasurementStore;
using FittedALEStageRecords =
    svmp::FE::systems::MeshNormalConstraintStageMeasurements;
using ConstFittedALEStageRecords =
    svmp::FE::systems::ConstMeshNormalConstraintStageMeasurements;

/// The fitted-ALE records of @p system (created on first use).
[[nodiscard]] inline FittedALEStageRecords fittedALERecords(
    svmp::FE::systems::FESystem& system)
{
  return svmp::FE::systems::meshNormalConstraintStageMeasurements(system);
}

/// The fitted-ALE records of @p system, read-only.
[[nodiscard]] inline ConstFittedALEStageRecords fittedALERecords(
    const svmp::FE::systems::FESystem& system) noexcept
{
  return svmp::FE::systems::meshNormalConstraintStageMeasurements(system);
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
