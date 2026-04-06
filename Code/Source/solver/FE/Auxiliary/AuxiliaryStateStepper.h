#ifndef SVMP_FE_AUXILIARY_STATE_STEPPER_H
#define SVMP_FE_AUXILIARY_STATE_STEPPER_H

/**
 * @file AuxiliaryStateStepper.h
 * @brief Time-stepping methods for auxiliary state blocks.
 *
 * Provides a `AuxiliaryStateStepper` abstract interface and concrete
 * implementations for advancing auxiliary state through a residual-based
 * model interface.
 *
 * ## Stepper implementations (phase 4)
 *
 * - `ForwardEulerStepper` — explicit Euler (no Jacobian needed).
 * - `BackwardEulerStepper` — implicit Euler with Newton solve.
 * - `BDF2Stepper` — 2nd-order BDF with Newton solve and history.
 * - `RK4Stepper` — classical 4th-order Runge-Kutta.
 *
 * ## Substepping
 *
 * Each stepper supports block-local substepping: the caller provides the
 * outer PDE step (t, dt) and a substep count.  The stepper takes
 * `substep_count` internal steps of size `dt/substep_count`.
 *
 * ## Workspace reuse
 *
 * Steppers maintain internal scratch vectors sized at setup time.
 * No per-call allocation.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Auxiliary/AuxiliaryStateModel.h"
#include "Auxiliary/AuxiliaryDerivativeProvider.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Stepper result
// ---------------------------------------------------------------------------

/**
 * @brief Result of a stepper advance call.
 */
struct AuxiliaryStepResult {
    /// Whether the step succeeded.
    bool converged{true};

    /// Number of Newton iterations (for implicit methods).
    int newton_iters{0};

    /// Number of substeps taken.
    int substeps_taken{1};

    /// Final residual norm (for implicit methods).
    Real final_residual_norm{0.0};
};

// ---------------------------------------------------------------------------
//  Abstract stepper interface
// ---------------------------------------------------------------------------

/**
 * @brief Abstract time-stepper for auxiliary state blocks.
 */
class AuxiliaryStateStepper {
public:
    virtual ~AuxiliaryStateStepper() = default;

    /// Human-readable method name.
    [[nodiscard]] virtual std::string methodName() const = 0;

    /// Whether this stepper requires Jacobians.
    [[nodiscard]] virtual bool requiresJacobian() const = 0;

    /// Whether this stepper requires history.
    [[nodiscard]] virtual int requiredHistoryDepth() const { return 0; }

    /**
     * @brief Set up the stepper for a given block dimension.
     *
     * Pre-allocates scratch vectors.
     */
    virtual void setup(int dimension, const AuxiliaryStepperSpec& spec);

    /**
     * @brief Advance the state by one outer step (possibly with substeps).
     *
     * @param model     The auxiliary model.
     * @param deriv     Derivative provider (for implicit methods).
     * @param x         Current state [in/out]: updated in place.
     * @param x_committed Committed state at start of outer step.
     * @param history   History views (for multi-step methods).
     * @param inputs    Auxiliary input values.
     * @param params    Parameters.
     * @param t         Current time at start of step.
     * @param dt        Outer PDE time step.
     * @param substep_count Number of local substeps (1 = no substepping).
     * @param entity_index  Entity index for per-entity models (default 0).
     *
     * @return Step result (convergence info).
     */
    virtual AuxiliaryStepResult advance(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        std::span<Real> x,
        std::span<const Real> x_committed,
        std::span<const std::span<const Real>> history,
        std::span<const Real> inputs,
        std::span<const Real> params,
        Real t,
        Real dt,
        int substep_count = 1,
        std::size_t entity_index = 0) = 0;

    /**
     * @brief Advance from the current work state (for multirate dispatch).
     *
     * Unlike advance(), this does NOT reset x from x_committed.  The
     * caller provides x already set to the intermediate work state from
     * a previous substep.  x_prev is the state at the START of this
     * substep (used for xdot computation in implicit methods).
     *
     * Default implementation delegates to advance() with x_committed = x_prev.
     */
    virtual AuxiliaryStepResult advanceFromWork(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        std::span<Real> x,
        std::span<const Real> x_prev,
        std::span<const std::span<const Real>> history,
        std::span<const Real> inputs,
        std::span<const Real> params,
        Real t,
        Real dt_sub,
        std::size_t entity_index = 0);

protected:
    int dim_{0};
    AuxiliaryStepperSpec spec_{};
    bool is_setup_{false};
};

// ---------------------------------------------------------------------------
//  Forward Euler
// ---------------------------------------------------------------------------

class ForwardEulerStepper final : public AuxiliaryStateStepper {
public:
    [[nodiscard]] std::string methodName() const override { return "ForwardEuler"; }
    [[nodiscard]] bool requiresJacobian() const override { return false; }

    void setup(int dimension, const AuxiliaryStepperSpec& spec) override;

    AuxiliaryStepResult advance(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        std::span<Real> x,
        std::span<const Real> x_committed,
        std::span<const std::span<const Real>> history,
        std::span<const Real> inputs,
        std::span<const Real> params,
        Real t, Real dt, int substep_count, std::size_t entity_index) override;

private:
    std::vector<Real> scratch_residual_{};
    std::vector<Real> scratch_xdot_{};
};

// ---------------------------------------------------------------------------
//  Backward Euler
// ---------------------------------------------------------------------------

class BackwardEulerStepper final : public AuxiliaryStateStepper {
public:
    [[nodiscard]] std::string methodName() const override { return "BackwardEuler"; }
    [[nodiscard]] bool requiresJacobian() const override { return true; }

    void setup(int dimension, const AuxiliaryStepperSpec& spec) override;

    AuxiliaryStepResult advance(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        std::span<Real> x,
        std::span<const Real> x_committed,
        std::span<const std::span<const Real>> history,
        std::span<const Real> inputs,
        std::span<const Real> params,
        Real t, Real dt, int substep_count, std::size_t entity_index) override;

private:
    std::vector<Real> scratch_residual_{};
    std::vector<Real> scratch_xdot_{};
    std::vector<Real> scratch_dFdx_{};
    std::vector<Real> scratch_dFdxdot_{};
    std::vector<Real> scratch_delta_{};
};

// ---------------------------------------------------------------------------
//  RK4
// ---------------------------------------------------------------------------

class RK4Stepper final : public AuxiliaryStateStepper {
public:
    [[nodiscard]] std::string methodName() const override { return "RK4"; }
    [[nodiscard]] bool requiresJacobian() const override { return false; }

    void setup(int dimension, const AuxiliaryStepperSpec& spec) override;

    AuxiliaryStepResult advance(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        std::span<Real> x,
        std::span<const Real> x_committed,
        std::span<const std::span<const Real>> history,
        std::span<const Real> inputs,
        std::span<const Real> params,
        Real t, Real dt, int substep_count, std::size_t entity_index) override;

private:
    std::vector<Real> scratch_residual_{};
    std::vector<Real> scratch_xdot_{};
    std::vector<Real> scratch_k1_{};
    std::vector<Real> scratch_k2_{};
    std::vector<Real> scratch_k3_{};
    std::vector<Real> scratch_k4_{};
    std::vector<Real> scratch_x_stage_{};
};

// ---------------------------------------------------------------------------
//  BDF2
// ---------------------------------------------------------------------------

class BDF2Stepper final : public AuxiliaryStateStepper {
public:
    [[nodiscard]] std::string methodName() const override { return "BDF2"; }
    [[nodiscard]] bool requiresJacobian() const override { return true; }
    [[nodiscard]] int requiredHistoryDepth() const override { return 1; }

    void setup(int dimension, const AuxiliaryStepperSpec& spec) override;

    AuxiliaryStepResult advance(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        std::span<Real> x,
        std::span<const Real> x_committed,
        std::span<const std::span<const Real>> history,
        std::span<const Real> inputs,
        std::span<const Real> params,
        Real t, Real dt, int substep_count, std::size_t entity_index) override;

private:
    std::vector<Real> scratch_residual_{};
    std::vector<Real> scratch_xdot_{};
    std::vector<Real> scratch_dFdx_{};
    std::vector<Real> scratch_dFdxdot_{};
    std::vector<Real> scratch_delta_{};

    BackwardEulerStepper fallback_{};
};

// ---------------------------------------------------------------------------
//  Factory
// ---------------------------------------------------------------------------

/**
 * @brief Create a stepper by method name.
 *
 * Known names: "ForwardEuler", "BackwardEuler", "RK4", "BDF2".
 */
[[nodiscard]] std::unique_ptr<AuxiliaryStateStepper> createStepper(
    const std::string& method_name);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_STATE_STEPPER_H
