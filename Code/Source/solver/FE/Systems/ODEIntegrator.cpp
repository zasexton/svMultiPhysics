/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/ODEIntegrator.h"

#include "Core/FEException.h"

#include "Forms/PointEvaluator.h"
#include "Systems/AuxiliaryState.h"

#include <cmath>
#include <limits>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

struct ScopedAuxSlotValue final {
    AuxiliaryState& state;
    std::uint32_t slot{0};
    Real old_value{0.0};

    ScopedAuxSlotValue(AuxiliaryState& s, std::uint32_t slot_in, Real new_value)
        : state(s)
        , slot(slot_in)
    {
        auto vals = state.values();
        FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
                    "ODEIntegrator: auxiliary slot out of range");
        old_value = vals[slot];
        vals[slot] = new_value;
    }

    ~ScopedAuxSlotValue()
    {
        // Best-effort restore; do not throw from destructor.
        try {
            auto vals = state.values();
            if (slot < vals.size()) {
                vals[slot] = old_value;
            }
        } catch (...) {
        }
    }
};

[[nodiscard]] Real newtonSolveScalar(const std::function<std::pair<Real, Real>(Real)>& f_and_df,
                                    Real x0,
                                    int max_iters,
                                    Real tol_abs,
                                    Real tol_rel)
{
    Real x = x0;
    for (int it = 0; it < max_iters; ++it) {
        const auto [f, df] = f_and_df(x);
        const Real scale = tol_abs + tol_rel * (Real(1.0) + std::abs(x));
        if (std::abs(f) <= scale) {
            return x;
        }
        if (std::abs(df) < Real(1e-16)) {
            throw ConvergenceException("ODEIntegrator: Newton derivative near zero",
                                       it,
                                       std::abs(f),
                                       __FILE__,
                                       __LINE__);
        }
        const Real dx = -f / df;
        x += dx;
        if (std::abs(dx) <= scale) {
            return x;
        }
    }
    const auto [f, df] = f_and_df(x);
    (void)df;
    throw ConvergenceException("ODEIntegrator: Newton did not converge",
                               max_iters,
                               std::abs(f),
                               __FILE__,
                               __LINE__);
}

} // namespace

void ODEIntegrator::advance(ODEMethod method,
                            std::uint32_t state_slot,
                            AuxiliaryState& state,
                            const forms::FormExpr& rhs,
                            const std::optional<forms::FormExpr>& d_rhs_dX,
                            std::span<const Real> integrals,
                            std::span<const Real> params,
                            Real t,
                            Real dt)
{
    FE_THROW_IF(state_slot >= state.size(), InvalidArgumentException,
                "ODEIntegrator::advance: state_slot out of range");
    FE_THROW_IF(!rhs.isValid(), InvalidArgumentException,
                "ODEIntegrator::advance: invalid RHS expression");

    if (dt <= Real(0.0)) {
        return;
    }

    // Ensure the variable exists (and get previous/work value).
    auto vals = state.values();
    Real& x_ref = vals[state_slot];
    const Real x_prev = x_ref;

    auto eval_rhs = [&](Real x, Real t_eval) -> Real
    {
        ScopedAuxSlotValue set(state, state_slot, x);
        forms::PointEvalContext pctx;
        pctx.x = {0.0, 0.0, 0.0};
        pctx.time = t_eval;
        pctx.dt = dt;
        pctx.jit_constants = params;
        pctx.coupled_integrals = integrals;
        pctx.coupled_aux = state.values();
        return forms::evaluateScalarAt(rhs, pctx);
    };

    switch (method) {
        case ODEMethod::ForwardEuler: {
            const Real f = eval_rhs(x_prev, t);
            x_ref = x_prev + dt * f;
            return;
        }

        case ODEMethod::RK4: {
            const Real k1 = eval_rhs(x_prev, t);
            const Real k2 = eval_rhs(x_prev + Real(0.5) * dt * k1, t + Real(0.5) * dt);
            const Real k3 = eval_rhs(x_prev + Real(0.5) * dt * k2, t + Real(0.5) * dt);
            const Real k4 = eval_rhs(x_prev + dt * k3, t + dt);
            x_ref = x_prev + (dt / Real(6.0)) * (k1 + Real(2.0) * k2 + Real(2.0) * k3 + k4);
            return;
        }

        case ODEMethod::BackwardEuler: {
            FE_THROW_IF(!d_rhs_dX || !d_rhs_dX->isValid(), InvalidArgumentException,
                        "ODEIntegrator::advance(BackwardEuler): missing analytic d_rhs_dX expression");
            const Real x_guess = x_prev + dt * eval_rhs(x_prev, t);

            auto eval_drhs = [&](Real x, Real t_eval) -> Real {
                ScopedAuxSlotValue set(state, state_slot, x);
                forms::PointEvalContext pctx;
                pctx.x = {0.0, 0.0, 0.0};
                pctx.time = t_eval;
                pctx.dt = dt;
                pctx.jit_constants = params;
                pctx.coupled_integrals = integrals;
                pctx.coupled_aux = state.values();
                return forms::evaluateScalarAt(*d_rhs_dX, pctx);
            };

            auto f_and_df = [&](Real x) -> std::pair<Real, Real> {
                const Real fx = eval_rhs(x, t);
                const Real F = x - x_prev - dt * fx;
                const Real dfdx = eval_drhs(x, t);
                const Real dF = Real(1.0) - dt * dfdx;
                return {F, dF};
            };

            const Real x_new = newtonSolveScalar(f_and_df, x_guess, /*max_iters=*/50,
                                                 /*tol_abs=*/Real(1e-12),
                                                 /*tol_rel=*/Real(1e-10));
            x_ref = x_new;
            return;
        }

        case ODEMethod::BDF2: {
            // Requires two prior committed states. If unavailable, fall back to Backward Euler.
            if (!state.hasHistory(/*steps_back=*/2)) {
                advance(ODEMethod::BackwardEuler, state_slot, state, rhs, d_rhs_dX, integrals, params, t, dt);
                return;
            }

            FE_THROW_IF(!d_rhs_dX || !d_rhs_dX->isValid(), InvalidArgumentException,
                        "ODEIntegrator::advance(BDF2): missing analytic d_rhs_dX expression");

            const Real x_n = x_prev;
            const auto hist = state.previous(/*steps_back=*/2);
            FE_THROW_IF(state_slot >= hist.size(), InvalidArgumentException,
                        "ODEIntegrator::advance(BDF2): state history slot out of range");
            const Real x_nm1 = hist[state_slot];
            const Real x_guess = x_n + dt * eval_rhs(x_n, t);

            auto eval_drhs = [&](Real x, Real t_eval) -> Real {
                ScopedAuxSlotValue set(state, state_slot, x);
                forms::PointEvalContext pctx;
                pctx.x = {0.0, 0.0, 0.0};
                pctx.time = t_eval;
                pctx.dt = dt;
                pctx.jit_constants = params;
                pctx.coupled_integrals = integrals;
                pctx.coupled_aux = state.values();
                return forms::evaluateScalarAt(*d_rhs_dX, pctx);
            };

            auto f_and_df = [&](Real x) -> std::pair<Real, Real> {
                const Real fx = eval_rhs(x, t);
                // (3 x - 4 x_n + x_{n-1})/(2 dt) - f(x,t) = 0  (constant dt BDF2)
                const Real F = (Real(3.0) * x - Real(4.0) * x_n + x_nm1) - Real(2.0) * dt * fx;
                const Real dfdx = eval_drhs(x, t);
                const Real dF = Real(3.0) - Real(2.0) * dt * dfdx;
                return {F, dF};
            };

            const Real x_new = newtonSolveScalar(f_and_df, x_guess, /*max_iters=*/50,
                                                 /*tol_abs=*/Real(1e-12),
                                                 /*tol_rel=*/Real(1e-10));
            x_ref = x_new;
            return;
        }
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
