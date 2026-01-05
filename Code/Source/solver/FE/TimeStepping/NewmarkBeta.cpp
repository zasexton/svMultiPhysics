/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/NewmarkBeta.h"

#include "Core/FEException.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace timestepping {

NewmarkBetaIntegrator::NewmarkBetaIntegrator(NewmarkBetaIntegratorOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(!std::isfinite(options_.beta) || !std::isfinite(options_.gamma),
                InvalidArgumentException,
                "NewmarkBetaIntegrator: parameters must be finite");
    FE_THROW_IF(!(options_.beta > 0.0), InvalidArgumentException,
                "NewmarkBetaIntegrator: beta must be > 0");
    FE_THROW_IF(!(options_.gamma > 0.0), InvalidArgumentException,
                "NewmarkBetaIntegrator: gamma must be > 0");
}

assembly::TimeIntegrationContext
NewmarkBetaIntegrator::buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const
{
    assembly::TimeIntegrationContext ctx;
    ctx.integrator_name = name();

    if (max_time_derivative_order <= 0) {
        return ctx;
    }

    FE_THROW_IF(max_time_derivative_order > maxSupportedDerivativeOrder(),
                InvalidArgumentException,
                "TimeIntegrator '" + name() + "' does not support dt(·," + std::to_string(max_time_derivative_order) + ")");

    const double dt = state.dt;
    FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), InvalidArgumentException,
                "TimeIntegrator '" + name() + "': dt must be finite and > 0");

    // This integrator expects two history slots to hold scheme-specific constants:
    // u_prev  -> dt2 constant term
    // u_prev2 -> dt1 constant term
    const int available_history = static_cast<int>(state.u_history.size());
    FE_THROW_IF(available_history < 2, InvalidArgumentException,
                "TimeIntegrator '" + name() + "': requires at least 2 history states");

    const double beta = options_.beta;
    const double gamma = options_.gamma;

    if (max_time_derivative_order >= 1) {
        assembly::TimeDerivativeStencil s;
        s.order = 1;
        s.a.assign(3, 0.0);
        s.a[0] = static_cast<Real>(gamma / (beta * dt));
        s.a[2] = static_cast<Real>(1.0);
        ctx.dt1 = s;
    }

    if (max_time_derivative_order >= 2) {
        assembly::TimeDerivativeStencil s;
        s.order = 2;
        s.a.assign(2, 0.0);
        s.a[0] = static_cast<Real>(1.0 / (beta * dt * dt));
        s.a[1] = static_cast<Real>(1.0);
        ctx.dt2 = s;
    }

    return ctx;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp

