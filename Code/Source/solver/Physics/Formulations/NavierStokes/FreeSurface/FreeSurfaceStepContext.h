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
 * (sim, opts.scheme, transient_scheme, transient,
 * velocity_extension_artifact_comm and the service functions).  A service
 * becomes an FE call once FE provides it.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRunSupport.h"

#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "Mesh/Core/MeshComm.h"

#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/**
 * @brief The FE system, solution history and backend factory of a run, as
 *        the driver's simulation components hold them.
 *
 * Converts implicitly from any object with these three members (the
 * Application's SimulationComponents) and converts on to
 * FreeSurfaceSystemHandle.  Valid while the referenced pointers are.
 */
struct FreeSurfaceRunComponents {
  template <class Components>
    requires std::convertible_to<
                 decltype((std::declval<Components&>().fe_system)),
                 const std::unique_ptr<svmp::FE::systems::FESystem>&> &&
             std::convertible_to<
                 decltype((std::declval<Components&>().time_history)),
                 const std::unique_ptr<svmp::FE::timestepping::TimeHistory>&> &&
             std::convertible_to<
                 decltype((std::declval<Components&>().backend)),
                 const std::unique_ptr<svmp::FE::backends::BackendFactory>&>
  FreeSurfaceRunComponents(Components& components) noexcept
      : fe_system(components.fe_system),
        time_history(components.time_history),
        backend(components.backend)
  {
  }

  const std::unique_ptr<svmp::FE::systems::FESystem>& fe_system;
  const std::unique_ptr<svmp::FE::timestepping::TimeHistory>& time_history;
  const std::unique_ptr<svmp::FE::backends::BackendFactory>& backend;
};

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

  /// The run's FE system, solution history and backend factory.
  FreeSurfaceRunComponents sim;
  /// The transient system the time loop solves.
  svmp::FE::systems::TransientSystem& transient;
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

  /// Revision of the accepted level-set velocity-extension maps of the run,
  /// absent without extension maps (the driver's velocity-extension
  /// registry).
  std::function<std::optional<std::uint64_t>()>
      current_velocity_extension_revision{};
};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
