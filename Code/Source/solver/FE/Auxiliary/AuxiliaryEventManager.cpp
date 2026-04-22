#include "Auxiliary/AuxiliaryEventManager.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace systems {

std::vector<DetectedEvent> AuxiliaryEventManager::detectEvents(
    const AuxiliaryStateModel& model,
    std::span<const Real> x_pre,
    std::span<const Real> x_post,
    Real t,
    Real dt,
    std::span<const Real> inputs,
    std::span<const Real> params,
    std::span<const std::span<const Real>> history,
    std::size_t entity_index)
{
    last_events_.clear();
    diagnostics_.clear();
    chattering_detected_ = false;

    if (!model.hasEventFunctions()) return {};

    const int n = model.dimension();
    std::vector<Real> xdot_zero(static_cast<std::size_t>(n), 0.0);

    // Evaluate event functions at pre-step state.
    AuxiliaryLocalContext ctx_pre;
    ctx_pre.time = t;
    ctx_pre.dt = dt;
    ctx_pre.effective_dt = dt;
    ctx_pre.x = x_pre;
    ctx_pre.xdot = xdot_zero;
    ctx_pre.history = history;
    ctx_pre.inputs = inputs;
    ctx_pre.params = params;
    ctx_pre.entity_index = entity_index;

    auto events_pre = model.evaluateEvents(ctx_pre);

    // Evaluate at post-step state.
    AuxiliaryLocalContext ctx_post;
    ctx_post.time = t + dt;
    ctx_post.dt = dt;
    ctx_post.effective_dt = dt;
    ctx_post.x = x_post;
    ctx_post.xdot = xdot_zero;
    ctx_post.history = history;
    ctx_post.inputs = inputs;
    ctx_post.params = params;
    ctx_post.entity_index = entity_index;

    auto events_post = model.evaluateEvents(ctx_post);

    const auto n_events = events_pre.values.size();
    if (n_events != events_post.values.size()) return {};

    // Resize chattering tracker.
    if (last_trigger_times_.size() < n_events) {
        last_trigger_times_.resize(n_events, -1e30);
    }

    // Detect sign changes.
    for (std::size_t i = 0; i < n_events; ++i) {
        const Real g_pre = events_pre.values[i];
        const Real g_post = events_post.values[i];

        if (g_pre * g_post < 0.0) {
            // Sign change detected.
            DetectedEvent ev;
            ev.event_index = static_cast<int>(i);
            ev.direction = (g_post > g_pre) ? +1 : -1;

            // Localize the event time.
            if (localization_policy_ == EventLocalizationPolicy::StepBoundary) {
                ev.event_time = t + dt;
                ev.localized = false;
            } else {
                // Bisection-based localization.
                Real t_lo = t;
                Real t_hi = t + dt;
                Real g_lo = g_pre;

                for (int iter = 0; iter < max_localization_iters_; ++iter) {
                    Real t_mid = 0.5 * (t_lo + t_hi);

                    // Linear interpolation of state at t_mid.
                    const Real alpha = (t_mid - t) / dt;
                    std::vector<Real> x_mid(static_cast<std::size_t>(n));
                    for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) {
                        x_mid[k] = (1.0 - alpha) * x_pre[k] + alpha * x_post[k];
                    }

                    AuxiliaryLocalContext ctx_mid;
                    ctx_mid.time = t_mid;
                    ctx_mid.dt = dt;
                    ctx_mid.effective_dt = dt;
                    ctx_mid.x = x_mid;
                    ctx_mid.xdot = xdot_zero;
                    ctx_mid.history = history;
                    ctx_mid.inputs = inputs;
                    ctx_mid.params = params;
                    ctx_mid.entity_index = entity_index;

                    auto events_mid = model.evaluateEvents(ctx_mid);
                    const Real g_mid = events_mid.values[i];

                    if (g_lo * g_mid < 0.0) {
                        t_hi = t_mid;
                    } else {
                        t_lo = t_mid;
                        g_lo = g_mid;
                    }

                    if (t_hi - t_lo < 1e-14 * dt) break;
                }

                ev.event_time = 0.5 * (t_lo + t_hi);
                ev.localized = true;
            }

            // Chattering detection.
            const Real dt_since_last = ev.event_time - last_trigger_times_[i];
            if (dt_since_last > 0.0 && dt_since_last < chattering_threshold_) {
                chattering_detected_ = true;
                diagnostics_.push_back(
                    "Chattering detected on event " + std::to_string(i) +
                    ": dt_since_last=" + std::to_string(dt_since_last));
            }
            last_trigger_times_[i] = ev.event_time;

            last_events_.push_back(ev);
        }
    }

    return last_events_;
}

void AuxiliaryEventManager::applyTransition(
    const AuxiliaryStateModel& model,
    const DetectedEvent& event,
    std::span<Real> x,
    Real t,
    std::span<const Real> inputs,
    std::span<const Real> params,
    std::span<const std::span<const Real>> history,
    std::size_t entity_index,
    Real dt)
{
    const int n = model.dimension();
    std::vector<Real> xdot_zero(static_cast<std::size_t>(n), 0.0);

    AuxiliaryLocalContext ctx;
    ctx.time = t;
    ctx.dt = dt;
    ctx.effective_dt = dt;
    ctx.x = x;
    ctx.xdot = xdot_zero;
    ctx.history = history;
    ctx.inputs = inputs;
    ctx.params = params;
    ctx.entity_index = entity_index;

    model.resetAfterEvent(ctx, event.event_index, x);
}

} // namespace systems
} // namespace FE
} // namespace svmp
