/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_VSVO_BDF_CONTROLLER_H
#define SVMP_FE_TIMESTEPPING_VSVO_BDF_CONTROLLER_H

#include "TimeStepping/StepController.h"

namespace svmp {
namespace FE {
namespace timestepping {

struct VSVO_BDF_ControllerOptions {
    // Error control (weighted RMS): error_norm <= 1 means accept.
    double abs_tol{1e-8};
    double rel_tol{1e-6};

    int min_order{1};
    int max_order{5};
    int initial_order{1};

    int max_retries{10};

    double safety{0.9};
    double min_factor{0.2};
    double max_factor{5.0};

    double min_dt{0.0}; // 0 disables clamping
    double max_dt{0.0}; // 0 disables clamping

    // PI controller (applied as: fac = safety * err^{-pi_alpha/(p+1)} * prev_err^{-pi_beta/(p+1)}).
    double pi_alpha{0.7};
    double pi_beta{0.4};

    // Heuristics
    double increase_order_threshold{0.05};
    double nonlinear_decrease_factor{0.5};
};

/**
 * @brief Minimal variable-step/variable-order BDF controller (orders 1..5).
 *
 * This controller assumes `StepAttemptInfo::error_norm` is a weighted RMS norm
 * (<= 1 accepted) and uses a standard dt update:
 *   dt_{new} = dt * safety * error^{-1/(p+1)}
 *
 * It also proposes order increases when the error is very small.
 */
class VSVO_BDF_Controller final : public StepController {
public:
    explicit VSVO_BDF_Controller(VSVO_BDF_ControllerOptions options);

    [[nodiscard]] int maxRetries() const noexcept override { return options_.max_retries; }

    [[nodiscard]] StepDecision onAccepted(const StepAttemptInfo& info) override;
    [[nodiscard]] StepDecision onRejected(const StepAttemptInfo& info,
                                          StepRejectReason reason) override;

    [[nodiscard]] double absTol() const noexcept { return options_.abs_tol; }
    [[nodiscard]] double relTol() const noexcept { return options_.rel_tol; }
    [[nodiscard]] int minOrder() const noexcept { return options_.min_order; }
    [[nodiscard]] int maxOrder() const noexcept { return options_.max_order; }
    [[nodiscard]] int initialOrder() const noexcept { return options_.initial_order; }

private:
    [[nodiscard]] double clampDt(double dt) const noexcept;
    [[nodiscard]] int clampOrder(int order) const noexcept;

    VSVO_BDF_ControllerOptions options_{};
    double prev_error_norm_{-1.0};
    int prev_order_{0};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_VSVO_BDF_CONTROLLER_H
