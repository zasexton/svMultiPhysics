#include "Auxiliary/AuxiliaryStateStepper.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Base setup
// ---------------------------------------------------------------------------

void AuxiliaryStateStepper::setup(int dimension, const AuxiliaryStepperSpec& spec)
{
    dim_ = dimension;
    spec_ = spec;
    is_setup_ = true;
}

AuxiliaryStepResult AuxiliaryStateStepper::advanceFromWork(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    std::span<Real> x,
    std::span<const Real> x_prev,
    std::span<const std::span<const Real>> history,
    std::span<const Real> inputs,
    std::span<const Real> params,
    Real t,
    Real dt_sub,
    std::size_t entity_index)
{
    // Default: delegate to advance() with x_committed = x_prev,
    // substep_count = 1.  This resets x from x_prev (the intermediate
    // state from the previous substep), not from the original committed state.
    return advance(model, deriv, x, x_prev, history, inputs, params,
                   t, dt_sub, /*substep_count=*/1, entity_index);
}

// ---------------------------------------------------------------------------
//  Forward Euler
// ---------------------------------------------------------------------------

void ForwardEulerStepper::setup(int dimension, const AuxiliaryStepperSpec& spec)
{
    AuxiliaryStateStepper::setup(dimension, spec);
    const auto n = static_cast<std::size_t>(dimension);
    scratch_residual_.resize(n, 0.0);
    scratch_xdot_.resize(n, 0.0);
}

AuxiliaryStepResult ForwardEulerStepper::advance(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& /*deriv*/,
    std::span<Real> x,
    std::span<const Real> x_committed,
    std::span<const std::span<const Real>> history,
    std::span<const Real> inputs,
    std::span<const Real> params,
    Real t, Real dt, int substep_count, std::size_t entity_index)
{
    FE_THROW_IF(!is_setup_, InvalidStateException, "ForwardEulerStepper: not set up");
    const auto n = static_cast<std::size_t>(dim_);
    const int nsub = std::max(substep_count, 1);
    const Real h = dt / static_cast<Real>(nsub);

    // Start from committed state.
    std::copy(x_committed.begin(), x_committed.end(), x.begin());

    Real t_local = t;
    for (int s = 0; s < nsub; ++s) {
        // For explicit ODE: F(xdot, x, ...) = xdot - f(x, ...) = 0
        // So f(x) = -F(0, x) when xdot=0.
        std::fill(scratch_xdot_.begin(), scratch_xdot_.end(), 0.0);

        AuxiliaryLocalContext ctx;
        ctx.time = t_local;
        ctx.entity_index = entity_index;
        ctx.dt = dt;
        ctx.effective_dt = h;
        ctx.x = x;
        ctx.xdot = scratch_xdot_;
        ctx.history = history;
        ctx.inputs = inputs;
        ctx.params = params;

        AuxiliaryResidualRequest req;
        req.residual = scratch_residual_;
        model.evaluateResidual(ctx, req);

        // x_{n+1} = x_n + h * (-F(0, x_n))
        // For ODE row: F = xdot - rhs => -F = rhs when xdot=0
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += h * (-scratch_residual_[i]);
        }

        t_local += h;
    }

    return {/*converged=*/true, /*newton_iters=*/0, nsub, 0.0};
}

// ---------------------------------------------------------------------------
//  Backward Euler
// ---------------------------------------------------------------------------

void BackwardEulerStepper::setup(int dimension, const AuxiliaryStepperSpec& spec)
{
    AuxiliaryStateStepper::setup(dimension, spec);
    const auto n = static_cast<std::size_t>(dimension);
    const auto nn = n * n;
    scratch_residual_.resize(n, 0.0);
    scratch_xdot_.resize(n, 0.0);
    scratch_dFdx_.resize(nn, 0.0);
    scratch_dFdxdot_.resize(nn, 0.0);
    scratch_delta_.resize(n, 0.0);
}

AuxiliaryStepResult BackwardEulerStepper::advance(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    std::span<Real> x,
    std::span<const Real> x_committed,
    std::span<const std::span<const Real>> history,
    std::span<const Real> inputs,
    std::span<const Real> params,
    Real t, Real dt, int substep_count, std::size_t entity_index)
{
    FE_THROW_IF(!is_setup_, InvalidStateException, "BackwardEulerStepper: not set up");
    const auto n = static_cast<std::size_t>(dim_);
    const int nsub = std::max(substep_count, 1);
    const Real h = dt / static_cast<Real>(nsub);

    // Start from committed.
    std::copy(x_committed.begin(), x_committed.end(), x.begin());

    // Keep a copy of the "previous" state for each substep.
    std::vector<Real> x_prev(x.begin(), x.end());

    int total_newton = 0;
    Real final_norm = 0.0;
    Real t_local = t;

    for (int s = 0; s < nsub; ++s) {
        // Forward Euler prediction as initial guess.
        {
            std::fill(scratch_xdot_.begin(), scratch_xdot_.end(), 0.0);
            AuxiliaryLocalContext pred_ctx;
            pred_ctx.time = t_local; pred_ctx.dt = dt; pred_ctx.effective_dt = h;
            pred_ctx.x = x; pred_ctx.xdot = scratch_xdot_;
            pred_ctx.history = history; pred_ctx.inputs = inputs; pred_ctx.params = params;
            pred_ctx.entity_index = entity_index;
            AuxiliaryResidualRequest pred_req;
            pred_req.residual = scratch_residual_;
            model.evaluateResidual(pred_ctx, pred_req);
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = x_prev[i] + h * (-scratch_residual_[i]);
            }
        }

        // Newton iteration.
        bool converged = false;
        for (int it = 0; it < spec_.max_nonlinear_iters; ++it) {
            // Compute xdot = (x - x_prev) / h
            for (std::size_t i = 0; i < n; ++i) {
                scratch_xdot_[i] = (x[i] - x_prev[i]) / h;
            }

            AuxiliaryLocalContext ctx;
            ctx.time = t_local + h; ctx.dt = dt; ctx.effective_dt = h;
        ctx.entity_index = entity_index;
            ctx.x = x; ctx.xdot = scratch_xdot_;
            ctx.history = history; ctx.inputs = inputs; ctx.params = params;

            // Evaluate residual.
            AuxiliaryResidualRequest res_req;
            res_req.residual = scratch_residual_;
            model.evaluateResidual(ctx, res_req);

            // Check convergence.
            Real norm = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                norm += scratch_residual_[i] * scratch_residual_[i];
            }
            norm = std::sqrt(norm);
            final_norm = norm;

            const Real scale = spec_.nonlinear_tol_abs +
                               spec_.nonlinear_tol_rel * (1.0 + norm);
            if (norm <= scale) {
                converged = true;
                total_newton += it;
                break;
            }

            // Evaluate Jacobian: dF/dx + (1/h) * dF/d(xdot)
            AuxiliaryJacobianRequest jac_req;
            jac_req.dF_dx = scratch_dFdx_;
            jac_req.dF_dxdot = scratch_dFdxdot_;
            jac_req.want_dF_dxdot = true;
            jac_req.n = dim_;
            deriv.evaluateJacobian(model, ctx, jac_req);

            // Form effective Jacobian J = dF/dx + (1/h) * dF/dxdot
            // and solve J * delta = -F  (scalar Newton for n=1, dense for n>1).
            if (n == 1) {
                const Real J = scratch_dFdx_[0] + scratch_dFdxdot_[0] / h;
                if (std::abs(J) < 1e-30) break;
                x[0] -= scratch_residual_[0] / J;
            } else {
                // Dense LU-free: for small n, use Gaussian elimination.
                // Form J = dFdx + (1/h)*dFdxdot.
                std::vector<Real> J(n * n);
                for (std::size_t k = 0; k < n * n; ++k) {
                    J[k] = scratch_dFdx_[k] + scratch_dFdxdot_[k] / h;
                }

                // Simple Gauss elimination for small systems (n <= ~20).
                std::vector<Real> rhs(scratch_residual_.begin(),
                                       scratch_residual_.end());
                // Partial pivoting.
                std::vector<std::size_t> piv(n);
                for (std::size_t i = 0; i < n; ++i) piv[i] = i;

                for (std::size_t col = 0; col < n; ++col) {
                    // Find pivot.
                    std::size_t max_row = col;
                    Real max_val = std::abs(J[piv[col] * n + col]);
                    for (std::size_t row = col + 1; row < n; ++row) {
                        Real v = std::abs(J[piv[row] * n + col]);
                        if (v > max_val) { max_val = v; max_row = row; }
                    }
                    std::swap(piv[col], piv[max_row]);

                    const Real pivot = J[piv[col] * n + col];
                    if (std::abs(pivot) < 1e-30) break;

                    for (std::size_t row = col + 1; row < n; ++row) {
                        const Real factor = J[piv[row] * n + col] / pivot;
                        for (std::size_t k = col + 1; k < n; ++k) {
                            J[piv[row] * n + k] -= factor * J[piv[col] * n + k];
                        }
                        rhs[piv[row]] -= factor * rhs[piv[col]];
                    }
                }

                // Back substitution.
                std::vector<Real> delta(n, 0.0);
                for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
                    const auto r = static_cast<std::size_t>(row);
                    Real sum = rhs[piv[r]];
                    for (std::size_t k = r + 1; k < n; ++k) {
                        sum -= J[piv[r] * n + k] * delta[k];
                    }
                    const Real diag = J[piv[r] * n + r];
                    delta[r] = (std::abs(diag) > 1e-30) ? sum / diag : 0.0;
                }

                for (std::size_t i = 0; i < n; ++i) {
                    x[i] -= delta[i];
                }
            }
        }

        if (!converged) {
            return {false, total_newton + spec_.max_nonlinear_iters, s + 1, final_norm};
        }

        // Prepare for next substep.
        std::copy(x.begin(), x.end(), x_prev.begin());
        t_local += h;
    }

    return {true, total_newton, nsub, final_norm};
}

// ---------------------------------------------------------------------------
//  RK4
// ---------------------------------------------------------------------------

void RK4Stepper::setup(int dimension, const AuxiliaryStepperSpec& spec)
{
    AuxiliaryStateStepper::setup(dimension, spec);
    const auto n = static_cast<std::size_t>(dimension);
    scratch_residual_.resize(n, 0.0);
    scratch_xdot_.resize(n, 0.0);
    scratch_k1_.resize(n, 0.0);
    scratch_k2_.resize(n, 0.0);
    scratch_k3_.resize(n, 0.0);
    scratch_k4_.resize(n, 0.0);
    scratch_x_stage_.resize(n, 0.0);
}

AuxiliaryStepResult RK4Stepper::advance(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& /*deriv*/,
    std::span<Real> x,
    std::span<const Real> x_committed,
    std::span<const std::span<const Real>> history,
    std::span<const Real> inputs,
    std::span<const Real> params,
    Real t, Real dt, int substep_count, std::size_t entity_index)
{
    FE_THROW_IF(!is_setup_, InvalidStateException, "RK4Stepper: not set up");
    const auto n = static_cast<std::size_t>(dim_);
    const int nsub = std::max(substep_count, 1);
    const Real h = dt / static_cast<Real>(nsub);

    std::copy(x_committed.begin(), x_committed.end(), x.begin());
    std::fill(scratch_xdot_.begin(), scratch_xdot_.end(), 0.0);

    auto eval_rhs = [&](std::span<const Real> x_eval, Real t_eval,
                         std::span<Real> k_out) {
        AuxiliaryLocalContext ctx;
        ctx.time = t_eval; ctx.dt = dt; ctx.effective_dt = h;
        ctx.entity_index = entity_index;
        ctx.x = x_eval; ctx.xdot = scratch_xdot_;
        ctx.history = history; ctx.inputs = inputs; ctx.params = params;

        AuxiliaryResidualRequest req;
        req.residual = scratch_residual_;
        model.evaluateResidual(ctx, req);

        // f(x) = -F(0, x) for ODE rows.
        for (std::size_t i = 0; i < n; ++i) {
            k_out[i] = -scratch_residual_[i];
        }
    };

    Real t_local = t;
    for (int s = 0; s < nsub; ++s) {
        // k1
        eval_rhs(x, t_local, scratch_k1_);

        // k2
        for (std::size_t i = 0; i < n; ++i)
            scratch_x_stage_[i] = x[i] + 0.5 * h * scratch_k1_[i];
        eval_rhs(scratch_x_stage_, t_local + 0.5 * h, scratch_k2_);

        // k3
        for (std::size_t i = 0; i < n; ++i)
            scratch_x_stage_[i] = x[i] + 0.5 * h * scratch_k2_[i];
        eval_rhs(scratch_x_stage_, t_local + 0.5 * h, scratch_k3_);

        // k4
        for (std::size_t i = 0; i < n; ++i)
            scratch_x_stage_[i] = x[i] + h * scratch_k3_[i];
        eval_rhs(scratch_x_stage_, t_local + h, scratch_k4_);

        // Update
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += (h / 6.0) * (scratch_k1_[i] + 2.0 * scratch_k2_[i] +
                                   2.0 * scratch_k3_[i] + scratch_k4_[i]);
        }

        t_local += h;
    }

    return {true, 0, nsub, 0.0};
}

// ---------------------------------------------------------------------------
//  BDF2
// ---------------------------------------------------------------------------

void BDF2Stepper::setup(int dimension, const AuxiliaryStepperSpec& spec)
{
    AuxiliaryStateStepper::setup(dimension, spec);
    const auto n = static_cast<std::size_t>(dimension);
    const auto nn = n * n;
    scratch_residual_.resize(n, 0.0);
    scratch_xdot_.resize(n, 0.0);
    scratch_dFdx_.resize(nn, 0.0);
    scratch_dFdxdot_.resize(nn, 0.0);
    scratch_delta_.resize(n, 0.0);

    fallback_.setup(dimension, spec);
}

AuxiliaryStepResult BDF2Stepper::advance(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    std::span<Real> x,
    std::span<const Real> x_committed,
    std::span<const std::span<const Real>> history,
    std::span<const Real> inputs,
    std::span<const Real> params,
    Real t, Real dt, int substep_count, std::size_t entity_index)
{
    FE_THROW_IF(!is_setup_, InvalidStateException, "BDF2Stepper: not set up");

    // Need at least one history entry (x_{n-1}) for BDF2.
    // If unavailable, fall back to BackwardEuler.
    if (history.empty()) {
        return fallback_.advance(model, deriv, x, x_committed, history,
                                 inputs, params, t, dt, substep_count, entity_index);
    }

    const auto n = static_cast<std::size_t>(dim_);
    const Real h = dt; // BDF2 uses full step (no substepping for multi-step).

    // x_n = x_committed, x_{n-1} = history[0]
    const auto x_n = x_committed;
    const auto x_nm1 = history[0];

    // Initial guess: Forward Euler prediction.
    std::fill(scratch_xdot_.begin(), scratch_xdot_.end(), 0.0);
    {
        AuxiliaryLocalContext pred_ctx;
        pred_ctx.time = t; pred_ctx.dt = dt; pred_ctx.effective_dt = h;
        pred_ctx.x = x_committed; pred_ctx.xdot = scratch_xdot_;
        pred_ctx.history = history; pred_ctx.inputs = inputs; pred_ctx.params = params;
        pred_ctx.entity_index = entity_index;
        AuxiliaryResidualRequest pred_req;
        pred_req.residual = scratch_residual_;
        model.evaluateResidual(pred_ctx, pred_req);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = x_n[i] + h * (-scratch_residual_[i]);
        }
    }

    // Newton iteration for BDF2:
    // xdot = (3x - 4x_n + x_{n-1}) / (2h)
    // F(xdot, x, ...) = 0
    int total_newton = 0;
    Real final_norm = 0.0;
    bool converged = false;

    for (int it = 0; it < spec_.max_nonlinear_iters; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            scratch_xdot_[i] = (3.0 * x[i] - 4.0 * x_n[i] + x_nm1[i]) / (2.0 * h);
        }

        AuxiliaryLocalContext ctx;
        ctx.time = t + h; ctx.dt = dt; ctx.effective_dt = h;
        ctx.entity_index = entity_index;
        ctx.x = x; ctx.xdot = scratch_xdot_;
        ctx.history = history; ctx.inputs = inputs; ctx.params = params;

        AuxiliaryResidualRequest res_req;
        res_req.residual = scratch_residual_;
        model.evaluateResidual(ctx, res_req);

        Real norm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            norm += scratch_residual_[i] * scratch_residual_[i];
        norm = std::sqrt(norm);
        final_norm = norm;

        const Real scale = spec_.nonlinear_tol_abs +
                           spec_.nonlinear_tol_rel * (1.0 + norm);
        if (norm <= scale) {
            converged = true;
            total_newton = it;
            break;
        }

        AuxiliaryJacobianRequest jac_req;
        jac_req.dF_dx = scratch_dFdx_;
        jac_req.dF_dxdot = scratch_dFdxdot_;
        jac_req.want_dF_dxdot = true;
        jac_req.n = dim_;
        deriv.evaluateJacobian(model, ctx, jac_req);

        // J = dF/dx + (3/(2h)) * dF/dxdot
        const Real bdf2_coeff = 3.0 / (2.0 * h);
        if (n == 1) {
            const Real J = scratch_dFdx_[0] + bdf2_coeff * scratch_dFdxdot_[0];
            if (std::abs(J) < 1e-30) break;
            x[0] -= scratch_residual_[0] / J;
        } else {
            std::vector<Real> J(n * n);
            for (std::size_t k = 0; k < n * n; ++k) {
                J[k] = scratch_dFdx_[k] + bdf2_coeff * scratch_dFdxdot_[k];
            }

            // Gauss elimination (same as BackwardEuler).
            std::vector<Real> rhs(scratch_residual_.begin(), scratch_residual_.end());
            std::vector<std::size_t> piv(n);
            for (std::size_t i = 0; i < n; ++i) piv[i] = i;

            for (std::size_t col = 0; col < n; ++col) {
                std::size_t max_row = col;
                Real max_val = std::abs(J[piv[col] * n + col]);
                for (std::size_t row = col + 1; row < n; ++row) {
                    Real v = std::abs(J[piv[row] * n + col]);
                    if (v > max_val) { max_val = v; max_row = row; }
                }
                std::swap(piv[col], piv[max_row]);
                const Real pivot = J[piv[col] * n + col];
                if (std::abs(pivot) < 1e-30) break;
                for (std::size_t row = col + 1; row < n; ++row) {
                    const Real factor = J[piv[row] * n + col] / pivot;
                    for (std::size_t k = col + 1; k < n; ++k)
                        J[piv[row] * n + k] -= factor * J[piv[col] * n + k];
                    rhs[piv[row]] -= factor * rhs[piv[col]];
                }
            }

            std::vector<Real> delta(n, 0.0);
            for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
                const auto r = static_cast<std::size_t>(row);
                Real sum = rhs[piv[r]];
                for (std::size_t k = r + 1; k < n; ++k)
                    sum -= J[piv[r] * n + k] * delta[k];
                const Real diag = J[piv[r] * n + r];
                delta[r] = (std::abs(diag) > 1e-30) ? sum / diag : 0.0;
            }

            for (std::size_t i = 0; i < n; ++i)
                x[i] -= delta[i];
        }
    }

    return {converged, total_newton, 1, final_norm};
}

// ---------------------------------------------------------------------------
//  Factory
// ---------------------------------------------------------------------------

std::unique_ptr<AuxiliaryStateStepper> createStepper(const std::string& method_name)
{
    if (method_name == "ForwardEuler") return std::make_unique<ForwardEulerStepper>();
    if (method_name == "BackwardEuler") return std::make_unique<BackwardEulerStepper>();
    if (method_name == "RK4") return std::make_unique<RK4Stepper>();
    if (method_name == "BDF2") return std::make_unique<BDF2Stepper>();

    FE_THROW(InvalidArgumentException,
             "createStepper: unknown method '" + method_name + "'");
}

} // namespace systems
} // namespace FE
} // namespace svmp
