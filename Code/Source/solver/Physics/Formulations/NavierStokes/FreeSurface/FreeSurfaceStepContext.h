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
 * @brief What the free-surface step participants share with the transient
 *        driver: the FE system, the time-integration selection, the active
 *        FE communicator and the driver services they still call.
 *
 * Each participant derives from FreeSurfaceStepContext, so the code that
 * moved from the driver's transient loop keeps the names it used there
 * (sim, opts.scheme, transient_scheme, velocity_extension_artifact_comm and
 * the service functions).  A service becomes an FE call once FE provides it.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRunSupport.h"

#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <functional>
#include <string_view>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

struct FreeSurfaceStepContext {
  /// The time-loop options the participants read.
  struct TimeLoopSchemeOptions {
    svmp::FE::timestepping::SchemeKind scheme{
        svmp::FE::timestepping::SchemeKind::BackwardEuler};
  };

  /// The resolved transient time-integration selection.
  struct TransientScheme {
    svmp::FE::timestepping::SchemeKind scheme{
        svmp::FE::timestepping::SchemeKind::BackwardEuler};
    std::string_view canonical_name{};
  };

  /// The run's FE system.
  FreeSurfaceSystemHandle sim;
  TimeLoopSchemeOptions opts{};
  TransientScheme transient_scheme{};
  /// The active FE communicator of the run.
  svmp::MeshComm velocity_extension_artifact_comm{};

  /// Communicator-wide fingerprint of the content of a distributed FE vector
  /// (the driver's mesh-boundary state fingerprint).  Collective.
  std::function<std::uint64_t(
      const svmp::FE::backends::GenericVector& state,
      const svmp::MeshComm& comm)>
      collectiveMeshBoundaryStateFingerprint{};
};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
