#include "Systems/AuxiliaryInitializationSolver.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace systems {

InitializationResult AuxiliaryInitializationSolver::solve(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    std::span<Real> x,
    std::span<const Real> inputs,
    std::span<const Real> params,
    Real time,
    const InitializationOptions& opts,
    IndexReductionHook index_reduction)
{
    InitializationResult result;
    const int n = model.dimension();

    // Structural analysis.
    auto analysis = AuxiliaryDAEAnalyzer::analyze(model);

    if (analysis.n_algebraic == 0) {
        result.converged = true;
        result.diagnostics.push_back("Pure ODE — no initialization needed");
        return result;
    }

    // Optional index reduction.
    if (index_reduction) {
        if (index_reduction(model, x, analysis)) {
            result.diagnostics.push_back("Index reduction applied");
        }
    }

    // Try model's own initialization first.
    if (opts.prefer_model_initialization && model.hasConsistentInitialization()) {
        AuxiliaryInitializationRequest init_req;
        init_req.x = x;
        init_req.time = time;
        init_req.inputs = inputs;
        init_req.params = params;
        model.initializeAlgebraic(init_req);
        result.converged = true;
        result.diagnostics.push_back("Used model's initializeAlgebraic()");
        return result;
    }

    // Newton iteration on the full residual, updating only algebraic variables.
    const auto& alg_idx = analysis.algebraic_indices;
    const auto n_alg = static_cast<std::size_t>(analysis.n_algebraic);
    const auto nn = static_cast<std::size_t>(n);

    std::vector<Real> xdot(nn, 0.0); // xdot = 0 for initialization
    std::vector<Real> residual(nn, 0.0);
    std::vector<Real> dFdx(nn * nn, 0.0);

    for (int it = 0; it < opts.max_iterations; ++it) {
        // Evaluate residual.
        AuxiliaryLocalContext ctx;
        ctx.time = time;
        ctx.dt = 0.0;
        ctx.effective_dt = 0.0;
        ctx.x = x;
        ctx.xdot = xdot;
        ctx.inputs = inputs;
        ctx.params = params;

        AuxiliaryResidualRequest res_req;
        res_req.residual = residual;
        model.evaluateResidual(ctx, res_req);

        // Check convergence on algebraic rows only.
        Real alg_norm = 0.0;
        for (const auto idx : alg_idx) {
            alg_norm += residual[static_cast<std::size_t>(idx)] *
                        residual[static_cast<std::size_t>(idx)];
        }
        alg_norm = std::sqrt(alg_norm);
        result.final_residual_norm = alg_norm;

        const Real scale = opts.tol_abs + opts.tol_rel * (1.0 + alg_norm);
        if (alg_norm <= scale) {
            result.converged = true;
            result.iterations = it;
            result.diagnostics.push_back(
                "Converged in " + std::to_string(it) + " iterations, ||g|| = " +
                std::to_string(alg_norm));
            return result;
        }

        // Evaluate Jacobian.
        AuxiliaryJacobianRequest jac_req;
        jac_req.dF_dx = dFdx;
        jac_req.n = n;
        deriv.evaluateJacobian(model, ctx, jac_req);

        // Solve the algebraic subsystem: J_alg * delta_z = -g_alg
        // Extract the algebraic rows/columns of the Jacobian.
        if (n_alg == 1) {
            // Scalar case.
            const auto ai = static_cast<std::size_t>(alg_idx[0]);
            const Real J = dFdx[ai * nn + ai];
            if (std::abs(J) < 1e-30) {
                result.diagnostics.push_back("Singular Jacobian at algebraic variable");
                break;
            }
            x[ai] -= residual[ai] / J;
        } else {
            // Small dense solve via Gaussian elimination on the algebraic subblock.
            std::vector<Real> J_sub(n_alg * n_alg, 0.0);
            std::vector<Real> rhs_sub(n_alg, 0.0);

            for (std::size_t ri = 0; ri < n_alg; ++ri) {
                rhs_sub[ri] = residual[static_cast<std::size_t>(alg_idx[ri])];
                for (std::size_t ci = 0; ci < n_alg; ++ci) {
                    J_sub[ri * n_alg + ci] =
                        dFdx[static_cast<std::size_t>(alg_idx[ri]) * nn +
                             static_cast<std::size_t>(alg_idx[ci])];
                }
            }

            // Gaussian elimination with partial pivoting.
            std::vector<std::size_t> piv(n_alg);
            for (std::size_t i = 0; i < n_alg; ++i) piv[i] = i;

            for (std::size_t col = 0; col < n_alg; ++col) {
                std::size_t max_row = col;
                Real max_val = std::abs(J_sub[piv[col] * n_alg + col]);
                for (std::size_t row = col + 1; row < n_alg; ++row) {
                    Real v = std::abs(J_sub[piv[row] * n_alg + col]);
                    if (v > max_val) { max_val = v; max_row = row; }
                }
                std::swap(piv[col], piv[max_row]);
                const Real pivot = J_sub[piv[col] * n_alg + col];
                if (std::abs(pivot) < 1e-30) break;
                for (std::size_t row = col + 1; row < n_alg; ++row) {
                    const Real factor = J_sub[piv[row] * n_alg + col] / pivot;
                    for (std::size_t k = col + 1; k < n_alg; ++k)
                        J_sub[piv[row] * n_alg + k] -= factor * J_sub[piv[col] * n_alg + k];
                    rhs_sub[piv[row]] -= factor * rhs_sub[piv[col]];
                }
            }

            std::vector<Real> delta(n_alg, 0.0);
            for (int row = static_cast<int>(n_alg) - 1; row >= 0; --row) {
                const auto r = static_cast<std::size_t>(row);
                Real sum = rhs_sub[piv[r]];
                for (std::size_t k = r + 1; k < n_alg; ++k)
                    sum -= J_sub[piv[r] * n_alg + k] * delta[k];
                const Real diag = J_sub[piv[r] * n_alg + r];
                delta[r] = (std::abs(diag) > 1e-30) ? sum / diag : 0.0;
            }

            for (std::size_t i = 0; i < n_alg; ++i) {
                x[static_cast<std::size_t>(alg_idx[i])] -= delta[i];
            }
        }
    }

    if (!result.converged) {
        result.iterations = opts.max_iterations;
        result.diagnostics.push_back(
            "Did not converge in " + std::to_string(opts.max_iterations) +
            " iterations, ||g|| = " + std::to_string(result.final_residual_norm));
    }

    return result;
}

} // namespace systems
} // namespace FE
} // namespace svmp
