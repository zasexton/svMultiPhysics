/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_PRECONDITIONER_REUSE_POLICY_H
#define SVMP_FE_BACKENDS_PRECONDITIONER_REUSE_POLICY_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief Break-even refresh rule for a reusable preconditioner or factorization.
 *
 * A preconditioner built from the Jacobian of one solve may be applied to the
 * Jacobians of later solves (later Newton iterations, outer passes or time
 * steps with the same sparsity).  A stale preconditioner costs extra Krylov
 * iterations; a refresh costs one setup.  The rule refreshes as soon as the
 * extra Krylov work accumulated since the last refresh reaches the work of one
 * refresh:
 *
 *     sum_j max(0, n_j - n*_j)  >=  R,
 *
 * where n_j are the iteration counts of the solves since the refresh, n*_j the
 * iterations a fresh preconditioner would need for the residual reduction of
 * solve j, and R the setup cost expressed in Krylov iterations: setup work
 * divided by the work of one preconditioned iteration of the fresh solve.  The
 * backend supplies R with the fresh solve (FSILS: operation counts; Eigen:
 * measured times).  n*_j uses the convergence rate of the fresh solve,
 * log(r/r0) per iteration, applied to the reduction of solve j; this keeps
 * solves that stop early on the absolute tolerance (or late on a strict
 * relative one) from being mistaken for staleness.
 *
 * The same budget bounds a solve that uses a reused preconditioner: once it
 * exceeds n*_j plus the remaining allowance R - sum(excess), refreshing is
 * cheaper than continuing, so the caller abandons it, refreshes and repeats
 * the solve (staleIterationBudget()).
 *
 * This is the deterministic break-even (ski-rental) strategy: whatever the
 * future sequence of iteration counts, its total cost is at most twice the
 * cost of the best refresh schedule chosen with hindsight.  It contains no
 * tunable constant; R is derived from the operator and the preconditioner.
 *
 * Hard triggers always refresh: no preconditioner yet, a changed sparsity or
 * size, and a failed solve with a stale preconditioner (the caller refreshes
 * and repeats that solve).
 */
class PreconditionerReusePolicy {
public:
    enum class Reason : std::uint8_t {
        None,           ///< Reuse the current preconditioner.
        Initial,        ///< No preconditioner exists yet.
        Structure,      ///< Sparsity, size or block layout changed.
        Disabled,       ///< Reuse disabled: refresh every solve.
        BreakEven,      ///< Accumulated extra iterations reached the setup cost.
        StaleFailure    ///< The solve with a stale preconditioner failed.
    };

    struct Decision {
        bool refresh{true};
        Reason reason{Reason::Initial};
    };

    [[nodiscard]] static std::string_view reasonName(Reason reason) noexcept
    {
        switch (reason) {
            case Reason::None: return "reuse";
            case Reason::Initial: return "initial";
            case Reason::Structure: return "structure";
            case Reason::Disabled: return "disabled";
            case Reason::BreakEven: return "break_even";
            case Reason::StaleFailure: return "stale_failure";
        }
        return "unknown";
    }

    /// Decide whether the next solve needs a fresh preconditioner.
    [[nodiscard]] Decision beforeSolve(bool reuse_enabled, bool structure_changed) const noexcept
    {
        if (!valid_) {
            return {true, Reason::Initial};
        }
        if (structure_changed) {
            return {true, Reason::Structure};
        }
        if (!reuse_enabled) {
            return {true, Reason::Disabled};
        }
        if (excess_iterations_ >= refresh_cost_iterations_) {
            return {true, Reason::BreakEven};
        }
        return {false, Reason::None};
    }

    /// Record that a fresh preconditioner was built for the next solve.
    void recordRefresh() noexcept
    {
        valid_ = true;
        fresh_iterations_ = -1;
        excess_iterations_ = 0.0;
        refresh_cost_iterations_ = 0.0;
        log_rate_ = 0.0;
        ++refresh_count_;
    }

    /**
     * @brief Record a completed solve.
     * @param iterations Krylov iterations of the solve.
     * @param used_fresh_preconditioner true for the first solve after recordRefresh().
     * @param refresh_cost_iterations for a fresh solve: the setup cost in units of
     *        one preconditioned Krylov iteration of that solve (ignored otherwise).
     * @param residual_reduction final / initial residual norm of the solve
     *        (0 when unknown: iteration counts are then compared directly).
     */
    void recordSolve(int iterations,
                     bool used_fresh_preconditioner,
                     double refresh_cost_iterations = 0.0,
                     double residual_reduction = 0.0) noexcept
    {
        if (!valid_) {
            return;
        }
        const int n = std::max(0, iterations);
        if (used_fresh_preconditioner || fresh_iterations_ < 0) {
            fresh_iterations_ = n;
            refresh_cost_iterations_ =
                (std::isfinite(refresh_cost_iterations) && refresh_cost_iterations > 0.0)
                    ? refresh_cost_iterations
                    : 0.0;
            log_rate_ = (n > 0 && informativeReduction(residual_reduction))
                            ? std::log(residual_reduction) / static_cast<double>(n)
                            : 0.0;
            return;
        }
        excess_iterations_ +=
            std::max(0.0, static_cast<double>(n) - expectedFreshIterations(residual_reduction));
        ++reuse_count_;
    }

    /// Iterations a fresh preconditioner is expected to need for a residual
    /// reduction: from the convergence rate of the last fresh solve when known,
    /// else its iteration count.
    [[nodiscard]] double expectedFreshIterations(double residual_reduction) const noexcept
    {
        if (log_rate_ < 0.0 && informativeReduction(residual_reduction)) {
            return std::log(residual_reduction) / log_rate_;
        }
        return static_cast<double>(std::max(0, fresh_iterations_));
    }

    /// Iteration budget of a solve with the current (reused) preconditioner for
    /// a target residual reduction; 0 when no budget applies.
    [[nodiscard]] int staleIterationBudget(double target_reduction) const noexcept
    {
        if (!valid_ || fresh_iterations_ < 0) {
            return 0;
        }
        const double budget = expectedFreshIterations(target_reduction) +
                              std::max(0.0, refresh_cost_iterations_ - excess_iterations_);
        return std::max(1, static_cast<int>(std::ceil(budget)));
    }

    void invalidate() noexcept
    {
        valid_ = false;
        fresh_iterations_ = -1;
        excess_iterations_ = 0.0;
        refresh_cost_iterations_ = 0.0;
        log_rate_ = 0.0;
    }

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] int freshIterations() const noexcept { return fresh_iterations_; }
    [[nodiscard]] double excessIterations() const noexcept { return excess_iterations_; }
    [[nodiscard]] double refreshCostIterations() const noexcept { return refresh_cost_iterations_; }
    [[nodiscard]] std::uint64_t refreshCount() const noexcept { return refresh_count_; }
    [[nodiscard]] std::uint64_t reuseCount() const noexcept { return reuse_count_; }

private:
    [[nodiscard]] static bool informativeReduction(double r) noexcept
    {
        return std::isfinite(r) && r > 0.0 && r < 1.0;
    }

    bool valid_{false};
    double log_rate_{0.0};
    int fresh_iterations_{-1};
    double excess_iterations_{0.0};
    double refresh_cost_iterations_{0.0};
    std::uint64_t refresh_count_{0};
    std::uint64_t reuse_count_{0};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_PRECONDITIONER_REUSE_POLICY_H
