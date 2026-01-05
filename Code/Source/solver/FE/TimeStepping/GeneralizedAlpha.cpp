/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/GeneralizedAlpha.h"

#include "Core/FEException.h"
#include "Math/FiniteDifference.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace timestepping {

GeneralizedAlphaFirstOrderIntegrator::GeneralizedAlphaFirstOrderIntegrator(GeneralizedAlphaFirstOrderIntegratorOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(!std::isfinite(options_.alpha_m) || !std::isfinite(options_.alpha_f) || !std::isfinite(options_.gamma),
                InvalidArgumentException,
                "GeneralizedAlphaFirstOrderIntegrator: parameters must be finite");
    FE_THROW_IF(!(options_.alpha_f > 0.0), InvalidArgumentException,
                "GeneralizedAlphaFirstOrderIntegrator: alpha_f must be > 0");
    FE_THROW_IF(!(options_.gamma > 0.0), InvalidArgumentException,
                "GeneralizedAlphaFirstOrderIntegrator: gamma must be > 0");
    FE_THROW_IF(options_.history_rate_order < 1, InvalidArgumentException,
                "GeneralizedAlphaFirstOrderIntegrator: history_rate_order must be >= 1");
}

assembly::TimeIntegrationContext
GeneralizedAlphaFirstOrderIntegrator::buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const
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

    const double alpha_m = options_.alpha_m;
    const double alpha_f = options_.alpha_f;
    const double gamma = options_.gamma;

    FE_THROW_IF(!(alpha_f > 0.0), InvalidArgumentException,
                "TimeIntegrator '" + name() + "': alpha_f must be > 0");

    const int available_history = static_cast<int>(state.u_history.size());
    FE_THROW_IF(available_history < 2, InvalidArgumentException,
                "TimeIntegrator '" + name() + "': requires at least 2 history states (u^n and u^{n-1})");

    const int q_max = std::max(1, available_history - 1);
    const int q = std::max(1, std::min(options_.history_rate_order, q_max));

    const auto dt_hist = state.dt_history;
    const double dt_prev = (state.dt_prev > 0.0 && std::isfinite(state.dt_prev)) ? state.dt_prev : dt;

    auto historyDt = [&](int idx) -> double {
        if (idx < 0 || idx >= static_cast<int>(dt_hist.size())) {
            return dt_prev;
        }
        const double v = dt_hist[static_cast<std::size_t>(idx)];
        if (v > 0.0 && std::isfinite(v)) {
            return v;
        }
        return dt_prev;
    };

    std::vector<double> nodes;
    nodes.reserve(static_cast<std::size_t>(q + 1));
    nodes.push_back(0.0);
    double accum = 0.0;
    for (int j = 1; j <= q; ++j) {
        accum += historyDt(j - 1);
        nodes.push_back(-accum);
    }

    const auto w = math::finiteDifferenceWeights(/*derivative_order=*/1, /*x0=*/0.0, nodes);
    FE_THROW_IF(static_cast<int>(w.size()) != q + 1, InvalidArgumentException,
                "TimeIntegrator '" + name() + "': internal error computing history weights");

    const double c = alpha_m / (gamma * dt * alpha_f);
    const double c0 = (1.0 - alpha_m) - alpha_m * (1.0 - gamma) / gamma;

    assembly::TimeDerivativeStencil s;
    s.order = 1;
    s.a.assign(static_cast<std::size_t>(q + 2), 0.0);
    s.a[0] = static_cast<Real>(c);
    s.a[1] = static_cast<Real>(-c + c0 * w[0]);
    for (int j = 1; j <= q; ++j) {
        s.a[static_cast<std::size_t>(j + 1)] = static_cast<Real>(c0 * w[static_cast<std::size_t>(j)]);
    }

    ctx.dt1 = s;
    return ctx;
}

GeneralizedAlphaSecondOrderIntegrator::GeneralizedAlphaSecondOrderIntegrator(GeneralizedAlphaSecondOrderIntegratorOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(!std::isfinite(options_.alpha_m) || !std::isfinite(options_.alpha_f) ||
                    !std::isfinite(options_.beta) || !std::isfinite(options_.gamma),
                InvalidArgumentException,
                "GeneralizedAlphaSecondOrderIntegrator: parameters must be finite");
    FE_THROW_IF(!(options_.alpha_f > 0.0), InvalidArgumentException,
                "GeneralizedAlphaSecondOrderIntegrator: alpha_f must be > 0");
    FE_THROW_IF(!(options_.beta > 0.0), InvalidArgumentException,
                "GeneralizedAlphaSecondOrderIntegrator: beta must be > 0");
    FE_THROW_IF(!(options_.gamma > 0.0), InvalidArgumentException,
                "GeneralizedAlphaSecondOrderIntegrator: gamma must be > 0");
}

assembly::TimeIntegrationContext
GeneralizedAlphaSecondOrderIntegrator::buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const
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
    const double alpha_m = options_.alpha_m;
    const double alpha_f = options_.alpha_f;

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
        s.a[0] = static_cast<Real>(alpha_m / (alpha_f * beta * dt * dt));
        s.a[1] = static_cast<Real>(1.0);
        ctx.dt2 = s;
    }

    return ctx;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
