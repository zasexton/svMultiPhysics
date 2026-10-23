/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_STEPCONTEXT_H
#define SVMP_FE_TIMESTEPPING_STEPCONTEXT_H

/**
 * @file StepContext.h
 * @brief What step participants share with the time loop of a run: the FE
 *        system, solution history and backend factory, the time-integration
 *        selection, the active communicator and the host services they call.
 *
 * A participant may derive from StepContext, so that code moved out of a
 * host's transient loop keeps the names it used there (sim, transient,
 * opts.scheme, transient_scheme, active_comm and the service functions).
 * A service becomes an FE call once FE provides it.
 */

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Systems/FESystem.h"
#include "Systems/TransientSystem.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshComm.h"
#endif

#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace timestepping {

/**
 * @brief The FE system, solution history and backend factory of a run, as
 *        a host's simulation components hold them.
 *
 * Converts implicitly from any object with these three members, and on to
 * systems::FESystemHandle.  Valid while the referenced pointers are.
 */
struct RunComponents {
    template <class Components>
        requires std::convertible_to<
                     decltype((std::declval<Components&>().fe_system)),
                     const std::unique_ptr<systems::FESystem>&> &&
                 std::convertible_to<
                     decltype((std::declval<Components&>().time_history)),
                     const std::unique_ptr<TimeHistory>&> &&
                 std::convertible_to<
                     decltype((std::declval<Components&>().backend)),
                     const std::unique_ptr<backends::BackendFactory>&>
    RunComponents(Components& components) noexcept
        : fe_system(components.fe_system),
          time_history(components.time_history),
          backend(components.backend)
    {
    }

    const std::unique_ptr<systems::FESystem>& fe_system;
    const std::unique_ptr<TimeHistory>& time_history;
    const std::unique_ptr<backends::BackendFactory>& backend;
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
struct StepContext {
    /// The time-loop options the participants read.
    struct TimeLoopSchemeOptions {
        SchemeKind scheme{SchemeKind::BackwardEuler};
    };

    /// The resolved transient time-integration selection.
    struct TransientScheme {
        SchemeKind scheme{SchemeKind::BackwardEuler};
        std::string_view canonical_name{};
    };

    /// The run's FE system, solution history and backend factory.
    RunComponents sim;
    /// The transient system the time loop solves.
    systems::TransientSystem& transient;
    TimeLoopSchemeOptions opts{};
    TransientScheme transient_scheme{};
    /// The active FE communicator of the run.
    svmp::MeshComm active_comm{};

    /// Communicator-wide fingerprint of the content of a distributed FE
    /// vector (the host's mesh-boundary state fingerprint).  Collective.
    std::function<std::uint64_t(
        const backends::GenericVector& state,
        const svmp::MeshComm& comm)>
        collectiveMeshBoundaryStateFingerprint{};

    /// Revision of the accepted level-set velocity-extension maps of the run,
    /// absent without extension maps (the host's velocity-extension
    /// registry).
    std::function<std::optional<std::uint64_t>()>
        current_velocity_extension_revision{};
};
#endif

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_STEPCONTEXT_H
