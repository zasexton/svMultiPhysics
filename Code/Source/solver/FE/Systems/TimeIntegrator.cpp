/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/TimeIntegrator.h"

#include "Core/FEException.h"
#include "Math/FiniteDifference.h"

#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

assembly::TimeIntegrationContext
BackwardDifferenceIntegrator::buildContext(int max_time_derivative_order, const SystemStateView& state) const
{
    assembly::TimeIntegrationContext ctx;
    ctx.integrator_name = name();

    if (max_time_derivative_order <= 0) {
        return ctx;
    }

    FE_THROW_IF(max_time_derivative_order > maxSupportedDerivativeOrder(),
                InvalidArgumentException,
                "TimeIntegrator '" + name() + "' does not support dt(·," + std::to_string(max_time_derivative_order) + ")");

    const Real dt = static_cast<Real>(state.dt);
    FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), InvalidArgumentException,
                "TimeIntegrator '" + name() + "': dt must be finite and > 0");

    const int max_order = max_time_derivative_order;
    ctx.dt_extra.resize(max_order > 2 ? static_cast<std::size_t>(max_order - 2) : 0u);

    std::vector<double> nodes;
    nodes.reserve(static_cast<std::size_t>(max_order + 1));
    for (int j = 0; j <= max_order; ++j) {
        nodes.push_back(-static_cast<double>(dt) * static_cast<double>(j));
    }

    for (int k = 1; k <= max_order; ++k) {
        const auto w = math::finiteDifferenceWeights(/*derivative_order=*/k, /*x0=*/0.0,
                                                     std::span<const double>(nodes.data(), static_cast<std::size_t>(k + 1)));
        FE_THROW_IF(static_cast<int>(w.size()) != (k + 1), InvalidArgumentException,
                    "TimeIntegrator '" + name() + "': internal error computing dt(" + std::to_string(k) + ") weights");

        assembly::TimeDerivativeStencil s;
        s.order = k;
        s.a.resize(w.size());
        for (std::size_t j = 0; j < w.size(); ++j) {
            s.a[j] = static_cast<Real>(w[j]);
        }

        if (k == 1) {
            ctx.dt1 = s;
        } else if (k == 2) {
            ctx.dt2 = s;
        } else {
            ctx.dt_extra[static_cast<std::size_t>(k - 3)] = s;
        }
    }

    return ctx;
}

assembly::TimeIntegrationContext BDF2Integrator::buildContext(int max_time_derivative_order, const SystemStateView& state) const
{
    assembly::TimeIntegrationContext ctx;
    ctx.integrator_name = name();

    if (max_time_derivative_order <= 0) {
        return ctx;
    }

    FE_THROW_IF(max_time_derivative_order > maxSupportedDerivativeOrder(),
                InvalidArgumentException,
                "TimeIntegrator '" + name() + "' does not support dt(·," + std::to_string(max_time_derivative_order) + ")");

    const Real dt = static_cast<Real>(state.dt);
    FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), InvalidArgumentException,
                "TimeIntegrator '" + name() + "': dt must be finite and > 0");

    const Real dt_prev = static_cast<Real>(state.dt_prev > 0.0 ? state.dt_prev : state.dt);
    FE_THROW_IF(!(dt_prev > 0.0) || !std::isfinite(dt_prev), InvalidArgumentException,
                "TimeIntegrator '" + name() + "': dt_prev must be finite and > 0");

    const Real r = dt / dt_prev;
    FE_THROW_IF(!(r > 0.0) || !std::isfinite(r), InvalidArgumentException,
                "TimeIntegrator '" + name() + "': dt/dt_prev must be finite and > 0");

    assembly::TimeDerivativeStencil s;
    s.order = 1;
    const Real inv_dt = 1.0 / dt;
    s.a = {
        ((1.0 + 2.0 * r) / (1.0 + r)) * inv_dt,
        (-(1.0 + r)) * inv_dt,
        ((r * r) / (1.0 + r)) * inv_dt};
    ctx.dt1 = s;

    return ctx;
}

BDFIntegrator::BDFIntegrator(int order)
    : order_(order)
{
    FE_THROW_IF(order_ < 1 || order_ > 5, InvalidArgumentException,
                "BDFIntegrator: order must be in [1,5]");
}

assembly::TimeIntegrationContext BDFIntegrator::buildContext(int max_time_derivative_order, const SystemStateView& state) const
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

    const int available_history = static_cast<int>(state.u_history.size());
    const int max_stencil_points = order_ + max_time_derivative_order;
    const int required_history = max_stencil_points - 1;
    FE_THROW_IF(available_history < required_history, InvalidArgumentException,
                "TimeIntegrator '" + name() + "': requires at least " + std::to_string(required_history) +
                    " history states for dt order " + std::to_string(max_time_derivative_order));

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
    nodes.reserve(static_cast<std::size_t>(max_stencil_points));
    nodes.push_back(0.0);
    double accum = 0.0;
    for (int j = 1; j < max_stencil_points; ++j) {
        if (j == 1) {
            accum += dt;
        } else {
            accum += historyDt(j - 2);
        }
        nodes.push_back(-accum);
    }

    const int max_order = max_time_derivative_order;
    ctx.dt_extra.resize(max_order > 2 ? static_cast<std::size_t>(max_order - 2) : 0u);

    for (int k = 1; k <= max_order; ++k) {
        const int stencil_points_k = order_ + k;
        const auto w = math::finiteDifferenceWeights(/*derivative_order=*/k, /*x0=*/0.0,
                                                     std::span<const double>(nodes.data(), static_cast<std::size_t>(stencil_points_k)));
        FE_THROW_IF(static_cast<int>(w.size()) != stencil_points_k, InvalidArgumentException,
                    "TimeIntegrator '" + name() + "': internal error computing dt(" + std::to_string(k) + ") weights");

        assembly::TimeDerivativeStencil s;
        s.order = k;
        s.a.resize(w.size());
        for (std::size_t j = 0; j < w.size(); ++j) {
            s.a[j] = static_cast<Real>(w[j]);
        }

        if (k == 1) {
            ctx.dt1 = s;
        } else if (k == 2) {
            ctx.dt2 = s;
        } else {
            ctx.dt_extra[static_cast<std::size_t>(k - 3)] = s;
        }
    }

    return ctx;
}

} // namespace systems
} // namespace FE
} // namespace svmp
