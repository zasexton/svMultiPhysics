#ifndef SVMP_FE_AUXILIARY_EVENT_MANAGER_H
#define SVMP_FE_AUXILIARY_EVENT_MANAGER_H

/**
 * @file AuxiliaryEventManager.h
 * @brief Event detection, root bracketing, and state-transition management
 *        for auxiliary models with discontinuous or hybrid behavior.
 *
 * ## Event detection workflow
 *
 * 1. Model defines event functions g_i(x, t) via `evaluateEvents()`.
 * 2. After each stepper advance, event manager checks for sign changes.
 * 3. If a sign change is detected, root bracketing localizes the event.
 * 4. The model's `resetAfterEvent()` applies the state transition.
 * 5. Integration resumes from the post-event state.
 *
 * ## Chattering detection
 *
 * If the same event triggers repeatedly within a short interval,
 * the manager emits a diagnostic and can optionally:
 * - Reduce the time step
 * - Switch to a smoothed approximation
 * - Flag the event as chattering and skip further localization
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateModel.h"
#include "Auxiliary/AuxiliaryNonsmoothPolicy.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Event localization policy
// ---------------------------------------------------------------------------

/**
 * @brief Policy for localizing event crossings in time.
 */
enum class EventLocalizationPolicy : std::uint8_t {
    /// No localization; detect at step boundaries only.
    StepBoundary,

    /// Bisection root bracketing.
    Bisection,

    /// Illinois/Regula Falsi.
    RegulaFalsi,

    /// Brent's method.
    Brent
};

// ---------------------------------------------------------------------------
//  Event record
// ---------------------------------------------------------------------------

/**
 * @brief Record of a detected event.
 */
struct DetectedEvent {
    /// Index of the event function that triggered.
    int event_index{-1};

    /// Time of the event (after localization).
    Real event_time{0.0};

    /// Sign change direction: +1 = rising, -1 = falling.
    int direction{0};

    /// Whether the event was successfully localized.
    bool localized{false};
};

// ---------------------------------------------------------------------------
//  Event manager
// ---------------------------------------------------------------------------

/**
 * @brief Manages event detection and state transitions for auxiliary models.
 */
class AuxiliaryEventManager {
public:
    AuxiliaryEventManager() = default;

    // -----------------------------------------------------------------
    //  Configuration
    // -----------------------------------------------------------------

    /// Set the event localization policy.
    void setLocalizationPolicy(EventLocalizationPolicy policy) noexcept
    {
        localization_policy_ = policy;
    }

    /// Set the maximum number of root-bracketing iterations.
    void setMaxLocalizationIters(int n) noexcept
    {
        max_localization_iters_ = n;
    }

    /// Set the chattering detection threshold (minimum interval between
    /// repeated events on the same function).
    void setChatteringThreshold(Real dt_min) noexcept
    {
        chattering_threshold_ = dt_min;
    }

    /// Set the nonsmooth policy.
    void setNonsmoothPolicy(AuxiliaryNonsmoothPolicy policy) noexcept
    {
        nonsmooth_policy_ = policy;
    }

    // -----------------------------------------------------------------
    //  Event detection
    // -----------------------------------------------------------------

    /**
     * @brief Check for events after a step from t to t+dt.
     *
     * Evaluates event functions at the pre-step and post-step states.
     * Returns detected events (sign changes).
     *
     * @param model        The auxiliary model.
     * @param x_pre        State before the step.
     * @param x_post       State after the step.
     * @param t            Time at start of step.
     * @param dt           Step size.
     * @param inputs       Auxiliary inputs.
     * @param params       Parameters.
     * @param history      Entity-local auxiliary history.
     * @param entity_index Entity index for per-entity scoped models.
     */
    [[nodiscard]] std::vector<DetectedEvent> detectEvents(
        const AuxiliaryStateModel& model,
        std::span<const Real> x_pre,
        std::span<const Real> x_post,
        Real t,
        Real dt,
        std::span<const Real> inputs = {},
        std::span<const Real> params = {},
        std::span<const std::span<const Real>> history = {},
        std::size_t entity_index = 0);

    /**
     * @brief Apply state transition for a detected event.
     *
     * Calls the model's `resetAfterEvent()` to apply the transition.
     *
     * @param model    The model.
     * @param event    The detected event.
     * @param x        State vector [in/out] — updated by the transition.
     * @param t        Current time (at event).
     * @param inputs   Auxiliary inputs.
     * @param params   Parameters.
     * @param history  Entity-local auxiliary history.
     * @param entity_index Entity index for per-entity scoped models.
     * @param dt       Effective step size for the event transition context.
     */
    void applyTransition(
        const AuxiliaryStateModel& model,
        const DetectedEvent& event,
        std::span<Real> x,
        Real t,
        std::span<const Real> inputs = {},
        std::span<const Real> params = {},
        std::span<const std::span<const Real>> history = {},
        std::size_t entity_index = 0,
        Real dt = 0.0);

    // -----------------------------------------------------------------
    //  Diagnostics
    // -----------------------------------------------------------------

    /// Number of events detected in the last call.
    [[nodiscard]] std::size_t lastEventCount() const noexcept
    {
        return last_events_.size();
    }

    /// Last detected events.
    [[nodiscard]] const std::vector<DetectedEvent>& lastEvents() const noexcept
    {
        return last_events_;
    }

    /// Whether chattering was detected on any event function.
    [[nodiscard]] bool chatteringDetected() const noexcept
    {
        return chattering_detected_;
    }

    /// Diagnostic messages from the last detection/transition cycle.
    [[nodiscard]] const std::vector<std::string>& diagnostics() const noexcept
    {
        return diagnostics_;
    }

    /// Clear diagnostic state.
    void clearDiagnostics() noexcept
    {
        diagnostics_.clear();
        chattering_detected_ = false;
    }

private:
    EventLocalizationPolicy localization_policy_{EventLocalizationPolicy::Bisection};
    int max_localization_iters_{50};
    Real chattering_threshold_{1.0e-12};
    AuxiliaryNonsmoothPolicy nonsmooth_policy_{};

    std::vector<DetectedEvent> last_events_{};
    std::vector<std::string> diagnostics_{};
    bool chattering_detected_{false};

    /// Per-event-function: time of last trigger (for chattering detection).
    std::vector<Real> last_trigger_times_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_EVENT_MANAGER_H
