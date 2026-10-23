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
 * @brief Step participant of the fitted-ALE operator-stage normal
 *        measurements: the FE mesh-normal stage participant
 *        (FE/TimeStepping/MeshNormalStageParticipant.h) on the context of the
 *        free-surface step participants.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceStepContext.h"

#include "FE/Core/Types.h"
#include "FE/TimeStepping/MeshNormalStageParticipant.h"

#include <cstdint>
#include <string_view>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// Throws when fitted-ALE operator-stage measurements are declared
/// (@p declaration_count > 0) and the scheme is not supported.
inline void requireFittedALEOperatorStageSchemeCoverage(
    std::uint64_t declaration_count,
    bool scheme_supported,
    std::string_view scheme_name,
    int temporal_order)
{
  svmp::FE::timestepping::requireMeshNormalStageSchemeCoverage(
      declaration_count, scheme_supported, scheme_name, temporal_order);
}

/// The FE context of the mesh-normal stage participant: the run's FE
/// system, the loop scheme, the selected scheme's name, the active FE
/// communicator and the driver's mesh-boundary state fingerprint on it.
[[nodiscard]] svmp::FE::timestepping::MeshNormalStageContext
meshNormalStageContext(const FreeSurfaceStepContext& context);

class FittedALEStageParticipant final
    : public svmp::FE::timestepping::MeshNormalStageParticipant {
public:
  /// Checks the fitted-ALE declarations and scheme support on the active FE
  /// communicator (collective); throws when they differ across ranks or the
  /// scheme is unsupported.
  explicit FittedALEStageParticipant(const FreeSurfaceStepContext& context);
};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
