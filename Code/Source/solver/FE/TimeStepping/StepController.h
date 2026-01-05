/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_STEP_CONTROLLER_H
#define SVMP_FE_TIMESTEPPING_STEP_CONTROLLER_H

#include "TimeStepping/NewtonSolver.h"

#include <cstdint>
#include <string>

namespace svmp {
namespace FE {
namespace timestepping {

enum class StepRejectReason : std::uint8_t {
    NonlinearSolveFailed,
    ErrorTooLarge
};

struct StepAttemptInfo {
    double time{0.0};
    double t_end{0.0};
    double dt{0.0};
    double dt_prev{0.0};
    int step_index{0};
    int attempt_index{0};

    // Scheme-dependent (e.g., VSVO-BDF order). 0 means "not provided".
    int scheme_order{0};

    bool nonlinear_converged{true};
    NewtonReport newton{};

    // Optional error estimate (scheme-dependent); <= 0 means "not provided".
    double error_norm{-1.0};
};

struct StepDecision {
    bool accept{true};
    bool retry{false};
    double next_dt{0.0};
    int next_order{0};
    std::string message{};
};

/**
 * @brief Adaptive step-size policy interface for transient time loops.
 *
 * TimeLoop calls the controller after each attempt to decide whether to accept,
 * retry with a new dt, or continue with an updated dt for the next step.
 */
class StepController {
public:
    virtual ~StepController() = default;

    [[nodiscard]] virtual int maxRetries() const noexcept = 0;

    [[nodiscard]] virtual StepDecision onAccepted(const StepAttemptInfo& info) = 0;

    [[nodiscard]] virtual StepDecision onRejected(const StepAttemptInfo& info,
                                                  StepRejectReason reason) = 0;
};

struct SimpleStepControllerOptions {
    double min_dt{0.0};     ///< 0 disables clamping
    double max_dt{0.0};     ///< 0 disables clamping
    int max_retries{6};

    // Basic dt adaptation based on nonlinear iterations.
    double decrease_factor{0.5};
    double increase_factor{1.2};
    int target_newton_iterations{6};
};

/**
 * @brief Minimal adaptive step controller.
 *
 * - Rejects on nonlinear failure and reduces dt.
 * - On accept, optionally increases/decreases dt based on Newton iterations.
 *
 * This is intended as a safe default; higher-order controllers can be added
 * later (PI/PID, embedded error estimates, VSVO controllers, etc.).
 */
class SimpleStepController final : public StepController {
public:
    explicit SimpleStepController(SimpleStepControllerOptions options);

    [[nodiscard]] int maxRetries() const noexcept override { return options_.max_retries; }

    [[nodiscard]] StepDecision onAccepted(const StepAttemptInfo& info) override;

    [[nodiscard]] StepDecision onRejected(const StepAttemptInfo& info,
                                          StepRejectReason reason) override;

private:
    [[nodiscard]] double clamp(double dt) const noexcept;

    SimpleStepControllerOptions options_{};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_STEP_CONTROLLER_H
