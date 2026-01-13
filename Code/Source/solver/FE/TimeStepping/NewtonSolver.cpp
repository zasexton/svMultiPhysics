/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/NewtonSolver.h"

#include "Backends/Interfaces/BackendFactory.h"
#include "Constraints/AffineConstraints.h"
#include "Core/FEException.h"
#include "Systems/SystemsExceptions.h"

#if defined(FE_HAS_FSILS)
#  include "Backends/FSILS/FsilsVector.h"
#endif

#include <algorithm>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

namespace {

void axpy(backends::GenericVector& y, Real alpha, const backends::GenericVector& x)
{
    auto ys = y.localSpan();
    auto xs = x.localSpan();
    FE_CHECK_ARG(ys.size() == xs.size(), "NewtonSolver: axpy size mismatch");
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += alpha * xs[i];
    }
}

double residualNormForConvergence(const backends::GenericVector& r, backends::GenericVector& scratch)
{
    if (r.backendKind() != backends::BackendKind::FSILS) {
        return r.norm();
    }

#if defined(FE_HAS_FSILS)
    const auto* r_fs = dynamic_cast<const backends::FsilsVector*>(&r);
    auto* scratch_fs = dynamic_cast<backends::FsilsVector*>(&scratch);
    if (!r_fs || !scratch_fs) {
        return r.norm();
    }

    const auto src = r_fs->localSpan();
    auto dst = scratch_fs->localSpan();
    FE_CHECK_ARG(src.size() == dst.size(), "NewtonSolver: FSILS residual scratch size mismatch");
    std::copy(src.begin(), src.end(), dst.begin());

    // Assemble-time residuals are distributed by element ownership. FSILS expects overlap
    // contributions to be summed before norm/dot-based convergence checks.
    scratch_fs->accumulateOverlap();
    return scratch_fs->norm();
#else
    return r.norm();
#endif
}

} // namespace

NewtonSolver::NewtonSolver(NewtonOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(options_.max_iterations <= 0, InvalidArgumentException,
                "NewtonSolver: max_iterations must be > 0");
    FE_THROW_IF(options_.abs_tolerance < 0.0 || !std::isfinite(options_.abs_tolerance),
                InvalidArgumentException,
                "NewtonSolver: abs_tolerance must be finite and >= 0");
    FE_THROW_IF(options_.rel_tolerance < 0.0 || !std::isfinite(options_.rel_tolerance),
                InvalidArgumentException,
                "NewtonSolver: rel_tolerance must be finite and >= 0");
    FE_THROW_IF(options_.step_tolerance < 0.0 || !std::isfinite(options_.step_tolerance),
                InvalidArgumentException,
                "NewtonSolver: step_tolerance must be finite and >= 0");
}

systems::SystemStateView NewtonSolver::makeStateView(const TimeHistory& history, double solve_time) const
{
    systems::SystemStateView state;
    state.time = solve_time;
    state.dt = history.dt();
    state.dt_prev = history.dtPrev();
    state.u = history.uSpan();
    state.u_prev = history.uPrevSpan();
    state.u_prev2 = history.uPrev2Span();
    state.u_vector = &history.u();
    state.u_prev_vector = &history.uPrev();
    state.u_prev2_vector = &history.uPrev2();
    state.u_history = history.uHistorySpans();
    state.dt_history = history.dtHistory();
    return state;
}

void NewtonSolver::allocateWorkspace(const systems::FESystem& system,
                                     const backends::BackendFactory& factory,
                                     NewtonWorkspace& workspace) const
{
    const auto n_dofs = system.dofHandler().getNumDofs();
    FE_THROW_IF(n_dofs <= 0, systems::InvalidStateException, "NewtonSolver::allocateWorkspace: system has no DOFs");

    const auto* dist = system.distributedSparsityIfAvailable(options_.jacobian_op);
    if (dist != nullptr && factory.backendKind() != backends::BackendKind::Eigen) {
        workspace.jacobian = factory.createMatrix(*dist);
    } else {
        const auto& pattern = system.sparsity(options_.jacobian_op);
        workspace.jacobian = factory.createMatrix(pattern);
    }
    workspace.residual = factory.createVector(n_dofs);
    workspace.delta = factory.createVector(n_dofs);

    FE_CHECK_NOT_NULL(workspace.jacobian.get(), "NewtonSolver workspace.jacobian");
    FE_CHECK_NOT_NULL(workspace.residual.get(), "NewtonSolver workspace.residual");
    FE_CHECK_NOT_NULL(workspace.delta.get(), "NewtonSolver workspace.delta");
}

NewtonReport NewtonSolver::solveStep(systems::TransientSystem& transient,
                                     backends::LinearSolver& linear,
                                     double solve_time,
                                     TimeHistory& history,
                                     NewtonWorkspace& workspace,
                                     const backends::GenericVector* residual_addition) const
{
    FE_THROW_IF(!workspace.isAllocated(), InvalidArgumentException,
                "NewtonSolver::solveStep: workspace not allocated");

    auto& J = *workspace.jacobian;
    auto& r = *workspace.residual;
    auto& du = *workspace.delta;

    NewtonReport report;

    const auto& sys = transient.system();
    const auto& constraints = sys.constraints();

    history.updateGhosts();

    const systems::SystemStateView base_state = makeStateView(history, solve_time);
    FE_THROW_IF(!(base_state.dt > 0.0), InvalidArgumentException, "NewtonSolver: dt must be > 0");
    FE_THROW_IF(!std::isfinite(base_state.time), InvalidArgumentException, "NewtonSolver: solve_time must be finite");

    const int max_it = options_.max_iterations;
    for (int it = 0; it < max_it; ++it) {
        history.updateGhosts();

        if (!constraints.empty()) {
            constraints.distribute(history.u());
            history.u().updateGhosts();
        }

        systems::SystemStateView state = makeStateView(history, solve_time);

        auto J_view = J.createAssemblyView();
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");

        systems::AssemblyRequest req;
        const bool same_op = (options_.residual_op == options_.jacobian_op);
        if (same_op && options_.assemble_both_when_possible) {
            transient.system().beginTimeStep();
            req.op = options_.residual_op;
            req.want_matrix = true;
            req.want_vector = true;
            (void)transient.assemble(req, state, J_view.get(), r_view.get());
        } else {
            transient.system().beginTimeStep();
            req.op = options_.jacobian_op;
            req.want_matrix = true;
            (void)transient.assemble(req, state, J_view.get(), nullptr);

            req = systems::AssemblyRequest{};
            transient.system().beginTimeStep();
            req.op = options_.residual_op;
            req.want_vector = true;
            (void)transient.assemble(req, state, nullptr, r_view.get());
        }

        if (residual_addition != nullptr) {
            axpy(r, static_cast<Real>(1.0), *residual_addition);
        }

        if (!constraints.empty()) {
            std::vector<GlobalIndex> constrained;
            constrained.reserve(constraints.numConstraints());
            constraints.forEach([&constrained](const constraints::AffineConstraints::ConstraintView& cv) {
                if (cv.slave_dof >= 0) {
                    constrained.push_back(cv.slave_dof);
                }
            });

            auto r_zero = r.createAssemblyView();
            FE_CHECK_NOT_NULL(r_zero.get(), "NewtonSolver: residual zeroing view");
            r_zero->beginAssemblyPhase();
            r_zero->zeroVectorEntries(constrained);
            r_zero->finalizeAssembly();
        }

        report.residual_norm = residualNormForConvergence(r, du);
        if (it == 0) {
            report.residual_norm0 = report.residual_norm;
        }

        const bool abs_ok = report.residual_norm <= options_.abs_tolerance;
        const bool rel_ok = (options_.rel_tolerance <= 0.0)
            ? true
            : (report.residual_norm0 > 0.0
                   ? (report.residual_norm / report.residual_norm0 <= options_.rel_tolerance)
                   : abs_ok);
        if (abs_ok && rel_ok) {
            report.converged = true;
            report.iterations = it;
            return report;
        }

        du.zero();
        report.linear = linear.solve(J, du, r);
        FE_THROW_IF(!report.linear.converged, FEException,
                    "NewtonSolver: linear solve did not converge: " + report.linear.message);

        // Newton update: u <- u - du
        axpy(history.u(), static_cast<Real>(-1.0), du);
        if (!constraints.empty()) {
            constraints.distribute(history.u());
            history.u().updateGhosts();
        }

        if (options_.step_tolerance > 0.0) {
            const double step_norm = du.norm();
            if (step_norm <= options_.step_tolerance) {
                report.converged = true;
                report.iterations = it + 1;
                return report;
            }
        }
    }

    report.converged = false;
    report.iterations = max_it;
    return report;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
