/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/VSVO_BDF_Controller.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

VSVO_BDF_Controller::VSVO_BDF_Controller(VSVO_BDF_ControllerOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(options_.max_retries < 0, InvalidArgumentException,
                "VSVO_BDF_Controller: max_retries must be >= 0");
    FE_THROW_IF(options_.abs_tol <= 0.0 || !std::isfinite(options_.abs_tol), InvalidArgumentException,
                "VSVO_BDF_Controller: abs_tol must be finite and > 0");
    FE_THROW_IF(options_.rel_tol < 0.0 || !std::isfinite(options_.rel_tol), InvalidArgumentException,
                "VSVO_BDF_Controller: rel_tol must be finite and >= 0");
    FE_THROW_IF(options_.min_order < 1 || options_.max_order < options_.min_order, InvalidArgumentException,
                "VSVO_BDF_Controller: invalid order bounds");
    FE_THROW_IF(options_.initial_order < options_.min_order || options_.initial_order > options_.max_order,
                InvalidArgumentException,
                "VSVO_BDF_Controller: initial_order must be within [min_order,max_order]");
    FE_THROW_IF(!(options_.safety > 0.0) || !std::isfinite(options_.safety), InvalidArgumentException,
                "VSVO_BDF_Controller: safety must be finite and > 0");
    FE_THROW_IF(!(options_.min_factor > 0.0) || !std::isfinite(options_.min_factor), InvalidArgumentException,
                "VSVO_BDF_Controller: min_factor must be finite and > 0");
    FE_THROW_IF(!(options_.max_factor >= options_.min_factor) || !std::isfinite(options_.max_factor), InvalidArgumentException,
                "VSVO_BDF_Controller: max_factor must be finite and >= min_factor");
    FE_THROW_IF(options_.min_dt < 0.0 || !std::isfinite(options_.min_dt), InvalidArgumentException,
                "VSVO_BDF_Controller: min_dt must be finite and >= 0");
    FE_THROW_IF(options_.max_dt < 0.0 || !std::isfinite(options_.max_dt), InvalidArgumentException,
                "VSVO_BDF_Controller: max_dt must be finite and >= 0");
    FE_THROW_IF(options_.max_dt > 0.0 && options_.min_dt > 0.0 && options_.min_dt > options_.max_dt,
                InvalidArgumentException,
                "VSVO_BDF_Controller: min_dt must be <= max_dt");
    FE_THROW_IF(!(options_.pi_alpha >= 0.0) || !std::isfinite(options_.pi_alpha), InvalidArgumentException,
                "VSVO_BDF_Controller: pi_alpha must be finite and >= 0");
    FE_THROW_IF(!(options_.pi_beta >= 0.0) || !std::isfinite(options_.pi_beta), InvalidArgumentException,
                "VSVO_BDF_Controller: pi_beta must be finite and >= 0");
    FE_THROW_IF(options_.increase_order_threshold < 0.0 || !std::isfinite(options_.increase_order_threshold),
                InvalidArgumentException,
                "VSVO_BDF_Controller: increase_order_threshold must be finite and >= 0");
    FE_THROW_IF(!(options_.nonlinear_decrease_factor > 0.0 && options_.nonlinear_decrease_factor < 1.0) ||
                    !std::isfinite(options_.nonlinear_decrease_factor),
                InvalidArgumentException,
                "VSVO_BDF_Controller: nonlinear_decrease_factor must be finite and in (0,1)");
}

double VSVO_BDF_Controller::clampDt(double dt) const noexcept
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

int VSVO_BDF_Controller::clampOrder(int order) const noexcept
{
    return std::min(options_.max_order, std::max(options_.min_order, order));
}

StepDecision VSVO_BDF_Controller::onAccepted(const StepAttemptInfo& info)
{
    StepDecision d;
    const int p = clampOrder(info.scheme_order > 0 ? info.scheme_order : options_.initial_order);
    d.next_order = p;

    const double err = info.error_norm;
    if (!(err > 0.0) || !std::isfinite(err)) {
        d.accept = true;
        d.retry = false;
        d.next_dt = clampDt(info.dt);
        d.message = "accepted: no error estimate";
        return d;
    }

    d.accept = true;
    d.retry = false;

    if (err > 1.0) {
        d.accept = false;
        d.retry = (info.attempt_index < options_.max_retries);
        d.next_order = clampOrder(p - 1);

        // For rejected steps, base the reduction on the current error only. Using the
        // previous accepted error can (pathologically) prevent dt reduction when the
        // previous step was much easier than the current one.
        const double prev_err = err;
        const double inv_p1 = 1.0 / static_cast<double>(p + 1);
        const double k_i = options_.pi_alpha * inv_p1;
        const double k_p = options_.pi_beta * inv_p1;
        const double fac = options_.safety *
            std::pow(1.0 / std::max(err, 1e-16), k_i) *
            std::pow(1.0 / std::max(prev_err, 1e-16), k_p);

        const double fac_clamped = std::min(1.0, std::max(options_.min_factor, fac));
        d.next_dt = clampDt(info.dt * fac_clamped);
        d.message = "rejected: error too large";

        if (!(d.next_dt > 0.0) || !std::isfinite(d.next_dt)) {
            d.retry = false;
            d.message = "rejected: invalid dt after update";
        }
        return d;
    }

    auto candidateFactor = [&](int q, double err_q) -> double {
        if (!(err_q > 0.0) || !std::isfinite(err_q)) {
            return 1.0;
        }
        const bool have_prev = (info.step_index > 0) &&
            (prev_error_norm_ > 0.0 && std::isfinite(prev_error_norm_) && prev_order_ == q);
        const double prev_err = have_prev ? prev_error_norm_ : err_q;
        const double inv_q1 = 1.0 / static_cast<double>(q + 1);
        const double k_i = options_.pi_alpha * inv_q1;
        const double k_p = options_.pi_beta * inv_q1;
        const double fac = options_.safety *
            std::pow(1.0 / std::max(err_q, 1e-16), k_i) *
            std::pow(1.0 / std::max(prev_err, 1e-16), k_p);
        return std::min(options_.max_factor, std::max(options_.min_factor, fac));
    };

    struct Candidate {
        int order{0};
        double err{-1.0};
        double next_dt{0.0};
        double efficiency{-1.0};
    };

    std::vector<Candidate> candidates;
    candidates.reserve(3);

    auto addCandidate = [&](int q, double err_q) {
        if (q < options_.min_order || q > options_.max_order) {
            return;
        }
        if (!(err_q > 0.0) || !std::isfinite(err_q)) {
            return;
        }
        const double fac = candidateFactor(q, err_q);
        Candidate c;
        c.order = q;
        c.err = err_q;
        c.next_dt = clampDt(info.dt * fac);
        const double cost = static_cast<double>(q + 1);
        c.efficiency = (cost > 0.0) ? (c.next_dt / cost) : c.next_dt;
        candidates.push_back(c);
    };

    addCandidate(clampOrder(p - 1), info.error_norm_low);
    addCandidate(p, err);
    if (options_.increase_order_threshold <= 0.0 || info.error_norm_high < options_.increase_order_threshold) {
        addCandidate(clampOrder(p + 1), info.error_norm_high);
    }

    if (!candidates.empty()) {
        const auto best_it = std::max_element(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.efficiency < b.efficiency; });
        d.next_order = best_it->order;
        d.next_dt = best_it->next_dt;
    } else {
        const double fac = candidateFactor(p, err);
        d.next_dt = clampDt(info.dt * fac);
        d.next_order = p;
    }

    d.message = "accepted";

    prev_error_norm_ = err;
    prev_order_ = p;

    return d;
}

StepDecision VSVO_BDF_Controller::onRejected(const StepAttemptInfo& info, StepRejectReason reason)
{
    StepDecision d;
    d.accept = false;
    d.retry = (info.attempt_index < options_.max_retries);

    const int p = clampOrder(info.scheme_order > 0 ? info.scheme_order : options_.initial_order);
    d.next_order = clampOrder(p - 1);

    double next = info.dt;
    if (reason == StepRejectReason::NonlinearSolveFailed) {
        next *= options_.nonlinear_decrease_factor;
        d.message = "rejected: nonlinear solve failed";
    } else {
        const double err = info.error_norm;
        if (!(err > 0.0) || !std::isfinite(err)) {
            next *= options_.nonlinear_decrease_factor;
        } else {
            // Same reasoning as in onAccepted(): ensure a reduction is driven by the current error.
            const double prev_err = err;
            const double inv_p1 = 1.0 / static_cast<double>(p + 1);
            const double k_i = options_.pi_alpha * inv_p1;
            const double k_p = options_.pi_beta * inv_p1;
            const double fac = options_.safety *
                std::pow(1.0 / std::max(err, 1e-16), k_i) *
                std::pow(1.0 / std::max(prev_err, 1e-16), k_p);
            const double fac_clamped = std::min(1.0, std::max(options_.min_factor, fac));
            next *= fac_clamped;
        }
        d.message = "rejected: error too large";
    }
    d.next_dt = clampDt(next);

    if (!(d.next_dt > 0.0) || !std::isfinite(d.next_dt)) {
        d.retry = false;
        d.message = "rejected: invalid dt after update";
    }
    return d;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
