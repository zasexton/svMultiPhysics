/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SYSTEMSTATE_H
#define SVMP_FE_SYSTEMS_SYSTEMSTATE_H

#include "Core/Types.h"
#include "Core/ParameterValue.h"

#include <functional>
#include <optional>
#include <span>
#include <string_view>

namespace svmp {
namespace FE {

namespace backends {
class GenericVector;
} // namespace backends

namespace assembly {
struct TimeIntegrationContext;
}

namespace systems {

struct SystemStateView {
    double time{0.0};
    double dt{0.0};
    double dt_prev{0.0};

    std::span<const Real> u{};
    std::span<const Real> u_prev{};
    std::span<const Real> u_prev2{};

    // Optional backend vectors backing the spans above.
    // When provided, assemblers may use these to access global-indexed entries
    // even when the underlying local storage is not global-indexable (e.g., MPI layouts).
    const backends::GenericVector* u_vector{nullptr};
    const backends::GenericVector* u_prev_vector{nullptr};
    const backends::GenericVector* u_prev2_vector{nullptr};

    // Optional full history view for multistep methods.
    // Convention: u_history[k-1] corresponds to u^{n-k} (k=1 is u_prev).
    std::span<const std::span<const Real>> u_history{};

    // Optional time-step history aligned with u_history.
    // Convention: dt_history[k-1] corresponds to dt_{n-k+1} (k=1 is dt_prev).
    std::span<const double> dt_history{};

    // Optional transient time-integration context (set by Systems/TimeStepping).
    // If null, assembling forms containing `dt(...)` must fail.
    const assembly::TimeIntegrationContext* time_integration{nullptr};

    std::function<std::optional<Real>(std::string_view)> getRealParam{};
    std::function<std::optional<params::Value>(std::string_view)> getParam{};

    // Optional user-defined context pointer for advanced constitutive access.
    // Lifetime is managed by the caller.
    const void* user_data{nullptr};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMSTATE_H
