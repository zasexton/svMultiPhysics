#include "Systems/AuxiliaryDAEAnalyzer.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Structural analysis
// ---------------------------------------------------------------------------

DAEStructuralAnalysis AuxiliaryDAEAnalyzer::analyze(
    const AuxiliaryStateModel& model)
{
    DAEStructuralAnalysis result;
    const auto meta = model.structuralMetadata();
    const int n = model.dimension();

    for (int i = 0; i < n; ++i) {
        if (i < static_cast<int>(meta.variable_kinds.size()) &&
            meta.variable_kinds[static_cast<std::size_t>(i)] == AuxiliaryVariableKind::Algebraic) {
            result.algebraic_indices.push_back(i);
        } else {
            result.differential_indices.push_back(i);
        }
    }

    result.n_differential = static_cast<int>(result.differential_indices.size());
    result.n_algebraic = static_cast<int>(result.algebraic_indices.size());

    // Constraint groups from metadata.
    result.constraint_groups = meta.constraint_groups;
    if (result.constraint_groups.empty() && !result.algebraic_indices.empty()) {
        // Default: all algebraic variables in one group.
        result.constraint_groups.push_back(result.algebraic_indices);
    }

    // Index estimation.
    if (meta.dae_index_hint >= 0) {
        result.estimated_index = meta.dae_index_hint;
    } else if (result.n_algebraic == 0) {
        result.estimated_index = 0; // Pure ODE
    } else {
        result.estimated_index = 1; // Default assumption for mixed systems
    }

    result.structurally_nonsingular = true;

    if (result.n_algebraic > 0) {
        result.diagnostics.push_back(
            "Mixed DAE system: " + std::to_string(result.n_differential) +
            " differential + " + std::to_string(result.n_algebraic) +
            " algebraic variables, estimated index " +
            std::to_string(result.estimated_index));
    } else {
        result.diagnostics.push_back(
            "Pure ODE system: " + std::to_string(result.n_differential) +
            " differential variables");
    }

    return result;
}

// ---------------------------------------------------------------------------
//  Jacobian quality verification
// ---------------------------------------------------------------------------

JacobianQualityReport AuxiliaryDAEAnalyzer::verifyJacobian(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    const AuxiliaryLocalContext& ctx,
    Real fd_eps,
    Real tol)
{
    JacobianQualityReport report;
    const int n = model.dimension();
    const auto nn = static_cast<std::size_t>(n * n);

    if (!model.hasAnalyticJacobian()) {
        report.diagnostics.push_back("No analytic Jacobian — nothing to verify");
        return report;
    }

    // Evaluate analytic Jacobian.
    std::vector<Real> analytic_dFdx(nn, 0.0);
    AuxiliaryJacobianRequest analytic_req;
    analytic_req.dF_dx = analytic_dFdx;
    analytic_req.n = n;
    model.evaluateJacobian(ctx, analytic_req);

    // Evaluate FD Jacobian.
    std::vector<Real> fd_dFdx(nn, 0.0);
    std::vector<Real> base_residual(static_cast<std::size_t>(n), 0.0);
    std::vector<Real> pert_residual(static_cast<std::size_t>(n), 0.0);
    std::vector<Real> x_pert(ctx.x.begin(), ctx.x.end());

    AuxiliaryResidualRequest base_req;
    base_req.residual = base_residual;
    model.evaluateResidual(ctx, base_req);

    for (int j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const Real x_orig = x_pert[jj];
        const Real h = fd_eps * (1.0 + std::abs(x_orig));
        x_pert[jj] = x_orig + h;

        AuxiliaryLocalContext pert_ctx = ctx;
        pert_ctx.x = x_pert;

        AuxiliaryResidualRequest pert_req;
        pert_req.residual = pert_residual;
        model.evaluateResidual(pert_ctx, pert_req);

        for (int i = 0; i < n; ++i) {
            const auto idx = static_cast<std::size_t>(i * n + j);
            fd_dFdx[idx] = (pert_residual[static_cast<std::size_t>(i)] -
                             base_residual[static_cast<std::size_t>(i)]) / h;
        }

        x_pert[jj] = x_orig;
    }

    // Compare.
    report.abs_errors.resize(nn, 0.0);
    for (std::size_t k = 0; k < nn; ++k) {
        const Real abs_err = std::abs(analytic_dFdx[k] - fd_dFdx[k]);
        const Real scale = 1.0 + std::abs(fd_dFdx[k]);
        const Real rel_err = abs_err / scale;

        report.abs_errors[k] = abs_err;
        if (abs_err > report.max_abs_error) {
            report.max_abs_error = abs_err;
            report.worst_row = static_cast<int>(k) / n;
            report.worst_col = static_cast<int>(k) % n;
        }
        if (rel_err > report.max_rel_error) {
            report.max_rel_error = rel_err;
        }
    }

    report.consistent = (report.max_rel_error < tol);

    if (!report.consistent) {
        report.diagnostics.push_back(
            "Jacobian inconsistency: max_rel_error=" +
            std::to_string(report.max_rel_error) + " at (" +
            std::to_string(report.worst_row) + "," +
            std::to_string(report.worst_col) + "), tol=" +
            std::to_string(tol));
    }

    return report;
}

// ---------------------------------------------------------------------------
//  Scaling
// ---------------------------------------------------------------------------

DAEScalingRecommendation AuxiliaryDAEAnalyzer::computeScaling(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    const AuxiliaryLocalContext& ctx)
{
    const int n = model.dimension();
    DAEScalingRecommendation rec;
    rec.row_scales.assign(static_cast<std::size_t>(n), 1.0);
    rec.variable_scales.assign(static_cast<std::size_t>(n), 1.0);

    // Evaluate Jacobian.
    std::vector<Real> dFdx(static_cast<std::size_t>(n * n), 0.0);
    AuxiliaryJacobianRequest jreq;
    jreq.dF_dx = dFdx;
    jreq.n = n;
    deriv.evaluateJacobian(model, ctx, jreq);

    // Row scaling: 1 / max(|J_row|).
    for (int i = 0; i < n; ++i) {
        Real row_max = 0.0;
        for (int j = 0; j < n; ++j) {
            row_max = std::max(row_max,
                               std::abs(dFdx[static_cast<std::size_t>(i * n + j)]));
        }
        if (row_max > 1e-30) {
            rec.row_scales[static_cast<std::size_t>(i)] = 1.0 / row_max;
        }
    }

    // Variable scaling: 1 / max(|J_col|).
    for (int j = 0; j < n; ++j) {
        Real col_max = 0.0;
        for (int i = 0; i < n; ++i) {
            col_max = std::max(col_max,
                               std::abs(dFdx[static_cast<std::size_t>(i * n + j)]));
        }
        if (col_max > 1e-30) {
            rec.variable_scales[static_cast<std::size_t>(j)] = 1.0 / col_max;
        }
    }

    // Check if scaling is needed.
    rec.scaling_needed = false;
    for (int i = 0; i < n; ++i) {
        if (std::abs(rec.row_scales[static_cast<std::size_t>(i)] - 1.0) > 0.1 ||
            std::abs(rec.variable_scales[static_cast<std::size_t>(i)] - 1.0) > 0.1) {
            rec.scaling_needed = true;
            break;
        }
    }

    return rec;
}

// ---------------------------------------------------------------------------
//  Singularity check
// ---------------------------------------------------------------------------

bool AuxiliaryDAEAnalyzer::isStructurallySingular(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    const AuxiliaryLocalContext& ctx)
{
    const int n = model.dimension();
    std::vector<Real> dFdx(static_cast<std::size_t>(n * n), 0.0);
    AuxiliaryJacobianRequest jreq;
    jreq.dF_dx = dFdx;
    jreq.n = n;
    deriv.evaluateJacobian(model, ctx, jreq);

    // Check for zero rows or zero columns.
    for (int i = 0; i < n; ++i) {
        Real row_norm = 0.0;
        Real col_norm = 0.0;
        for (int j = 0; j < n; ++j) {
            row_norm += std::abs(dFdx[static_cast<std::size_t>(i * n + j)]);
            col_norm += std::abs(dFdx[static_cast<std::size_t>(j * n + i)]);
        }
        if (row_norm < 1e-30 || col_norm < 1e-30) {
            return true;
        }
    }

    return false;
}

} // namespace systems
} // namespace FE
} // namespace svmp
