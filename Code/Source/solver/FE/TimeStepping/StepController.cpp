/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/StepController.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace timestepping {

SimpleStepController::SimpleStepController(SimpleStepControllerOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(options_.max_retries < 0, InvalidArgumentException,
                "SimpleStepController: max_retries must be >= 0");
    FE_THROW_IF(!(options_.decrease_factor > 0.0) || options_.decrease_factor >= 1.0, InvalidArgumentException,
                "SimpleStepController: decrease_factor must be in (0,1)");
    FE_THROW_IF(!(options_.increase_factor >= 1.0), InvalidArgumentException,
                "SimpleStepController: increase_factor must be >= 1");
    FE_THROW_IF(options_.target_newton_iterations <= 0, InvalidArgumentException,
                "SimpleStepController: target_newton_iterations must be > 0");
    FE_THROW_IF(options_.min_dt < 0.0 || !std::isfinite(options_.min_dt), InvalidArgumentException,
                "SimpleStepController: min_dt must be finite and >= 0");
    FE_THROW_IF(options_.max_dt < 0.0 || !std::isfinite(options_.max_dt), InvalidArgumentException,
                "SimpleStepController: max_dt must be finite and >= 0");
    FE_THROW_IF(options_.max_dt > 0.0 && options_.min_dt > 0.0 && options_.min_dt > options_.max_dt,
                InvalidArgumentException,
                "SimpleStepController: min_dt must be <= max_dt");
}

double SimpleStepController::clamp(double dt) const noexcept
{
    double out = dt;
    if (options_.min_dt > 0.0) {
        out = std::max(out, options_.min_dt);
    }
    if (options_.max_dt > 0.0) {
        out = std::min(out, options_.max_dt);
    }
    return out;
}

StepDecision SimpleStepController::onAccepted(const StepAttemptInfo& info)
{
    StepDecision d;
    d.accept = true;
    d.retry = false;

    double next = info.dt;
    const int iters = info.newton.iterations;
    if (iters > options_.target_newton_iterations) {
        next *= options_.decrease_factor;
        d.message = "accepted: decreasing dt due to nonlinear iterations";
    } else if (iters < std::max(1, options_.target_newton_iterations / 2)) {
        next *= options_.increase_factor;
        d.message = "accepted: increasing dt due to fast nonlinear convergence";
    } else {
        d.message = "accepted";
    }

    d.next_dt = clamp(next);
    return d;
}

StepDecision SimpleStepController::onRejected(const StepAttemptInfo& info, StepRejectReason reason)
{
    StepDecision d;
    d.accept = false;
    d.retry = true;

    double next = info.dt;
    if (reason == StepRejectReason::NonlinearSolveFailed) {
        next *= options_.decrease_factor;
        d.message = "rejected: nonlinear solve failed";
    } else {
        next *= options_.decrease_factor;
        d.message = "rejected: error too large";
    }

    d.next_dt = clamp(next);
    if (!(d.next_dt > 0.0) || !std::isfinite(d.next_dt)) {
        d.retry = false;
        d.message = "rejected: invalid dt after update";
    }

    if (options_.min_dt > 0.0 && d.next_dt <= options_.min_dt + 0.0) {
        // If we're already at min_dt and still rejecting, stop retrying.
        if (info.attempt_index >= options_.max_retries) {
            d.retry = false;
        }
    }
    return d;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp

