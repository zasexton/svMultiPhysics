/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_TIME_HISTORY_H
#define SVMP_FE_TIMESTEPPING_TIME_HISTORY_H

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/Types.h"

#include <memory>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

/**
 * @brief Owns time-history solution vectors required by dt(·,k) stencils.
 *
 * Storage is backend-defined via backends::GenericVector. The vectors are sized
 * to the full system DOF count expected by FE/Assembly.
 */
class TimeHistory {
public:
    TimeHistory() = default;

    TimeHistory(std::unique_ptr<backends::GenericVector> u,
                std::unique_ptr<backends::GenericVector> u_prev,
                std::unique_ptr<backends::GenericVector> u_prev2);

    TimeHistory(std::unique_ptr<backends::GenericVector> u,
                std::vector<std::unique_ptr<backends::GenericVector>> history);

    TimeHistory(TimeHistory&&) noexcept = default;
    TimeHistory& operator=(TimeHistory&&) noexcept = default;

    TimeHistory(const TimeHistory&) = delete;
    TimeHistory& operator=(const TimeHistory&) = delete;

    [[nodiscard]] static TimeHistory allocate(const backends::BackendFactory& factory,
                                              GlobalIndex size,
                                              int history_depth = 2,
                                              bool allocate_second_order_state = false);

    [[nodiscard]] int historyDepth() const noexcept { return static_cast<int>(history_.size()); }

    [[nodiscard]] backends::GenericVector& u();
    [[nodiscard]] const backends::GenericVector& u() const;
    [[nodiscard]] backends::GenericVector& uPrev();
    [[nodiscard]] const backends::GenericVector& uPrev() const;
    [[nodiscard]] backends::GenericVector& uPrev2();
    [[nodiscard]] const backends::GenericVector& uPrev2() const;
    [[nodiscard]] backends::GenericVector& uPrevK(int k);
    [[nodiscard]] const backends::GenericVector& uPrevK(int k) const;

    [[nodiscard]] bool hasSecondOrderState() const noexcept { return static_cast<bool>(u_dot_) && static_cast<bool>(u_ddot_); }
    [[nodiscard]] bool hasUDotState() const noexcept { return static_cast<bool>(u_dot_); }
    [[nodiscard]] bool hasUDDotState() const noexcept { return static_cast<bool>(u_ddot_); }

    /**
     * @brief Ensure velocity/acceleration storage exists (allocates zero vectors if needed).
     *
     * Some second-order schemes (e.g., Newmark-β, structural generalized-α) store and
     * update (u̇,ü) alongside displacement history.
     */
    void ensureSecondOrderState(const backends::BackendFactory& factory);

    [[nodiscard]] backends::GenericVector& uDot();
    [[nodiscard]] const backends::GenericVector& uDot() const;
    [[nodiscard]] backends::GenericVector& uDDot();
    [[nodiscard]] const backends::GenericVector& uDDot() const;

    [[nodiscard]] std::span<Real> uSpan();
    [[nodiscard]] std::span<const Real> uSpan() const;
    [[nodiscard]] std::span<const Real> uPrevSpan() const;
    [[nodiscard]] std::span<const Real> uPrev2Span() const;
    [[nodiscard]] std::span<const Real> uPrevKSpan(int k) const;

    [[nodiscard]] std::span<Real> uDotSpan();
    [[nodiscard]] std::span<const Real> uDotSpan() const;
    [[nodiscard]] std::span<Real> uDDotSpan();
    [[nodiscard]] std::span<const Real> uDDotSpan() const;

    [[nodiscard]] std::span<const std::span<const Real>> uHistorySpans() const noexcept { return u_history_spans_; }
    [[nodiscard]] std::span<const double> dtHistory() const noexcept { return dt_history_; }

    void setTime(double t) noexcept { time_ = t; }
    [[nodiscard]] double time() const noexcept { return time_; }

    void setDt(double dt) noexcept { dt_ = dt; }
    [[nodiscard]] double dt() const noexcept { return dt_; }

    void setPrevDt(double dt_prev) noexcept
    {
        dt_prev_ = dt_prev;
        if (!dt_history_.empty()) {
            dt_history_[0] = dt_prev;
        }
    }
    [[nodiscard]] double dtPrev() const noexcept { return dt_prev_; }

    void setStepIndex(int step) noexcept { step_index_ = step; }
    [[nodiscard]] int stepIndex() const noexcept { return step_index_; }

    void updateGhosts();

    /**
     * @brief Fill unset dt history entries with a default value.
     *
     * Variable-step schemes use `dtHistory()`; callers that only set `dtPrev()`
     * (or use freshly-allocated histories) may have zeros in older slots.
     */
    void primeDtHistory(double dt_default) noexcept;

    /**
     * @brief Recreate history vectors using `factory` and copy values by global DOF index.
     *
     * Some backends (notably FSILS) require vectors to share the same internal
     * ordering/layout as the system matrix. Call this after the backend factory
     * has been "primed" by creating a matrix (e.g., the Jacobian) so subsequent
     * vectors use the correct backend layout.
     */
    void repack(const backends::BackendFactory& factory);

    /**
     * @brief Shift history on accepted step: u_prev2 <- u_prev, u_prev <- u.
     */
    void acceptStep(double accepted_dt);

    /**
     * @brief Set u to the current u_prev (common initial guess).
     */
    void resetCurrentToPrevious();

private:
    [[nodiscard]] static std::unique_ptr<backends::GenericVector>
    cloneLike(const backends::BackendFactory& factory, GlobalIndex size);

    void refreshHistoryViews();

    std::unique_ptr<backends::GenericVector> u_{};
    std::unique_ptr<backends::GenericVector> u_dot_{};
    std::unique_ptr<backends::GenericVector> u_ddot_{};
    std::vector<std::unique_ptr<backends::GenericVector>> history_{};
    std::vector<std::span<const Real>> u_history_spans_{};

    double time_{0.0};
    double dt_{0.0};
    double dt_prev_{0.0};
    std::vector<double> dt_history_{};
    int step_index_{0};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_TIME_HISTORY_H
