/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Eigen/EigenLinearSolver.h"

#include "Backends/Eigen/EigenMatrix.h"
#include "Backends/Eigen/EigenVector.h"
#include "Core/FEException.h"

#if defined(FE_HAS_EIGEN)
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/IterativeSolvers>
#endif

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

EigenLinearSolver::EigenLinearSolver(const SolverOptions& options)
{
    setOptions(options);
}

void EigenLinearSolver::setOptions(const SolverOptions& options)
{
    FE_THROW_IF(options.max_iter <= 0, InvalidArgumentException, "EigenLinearSolver: max_iter must be > 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "EigenLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "EigenLinearSolver: abs_tol must be >= 0");
    options_ = options;
}

namespace {

[[nodiscard]] bool check_convergence(const SolverReport& report, const SolverOptions& options) noexcept
{
    const bool has_abs = options.abs_tol > 0.0;
    const bool has_rel = options.rel_tol > 0.0;

    if (!has_abs && !has_rel) {
        return true;
    }

    const bool abs_ok = has_abs && (report.final_residual_norm <= options.abs_tol);
    const bool rel_ok = has_rel && (report.relative_residual <= options.rel_tol);
    return abs_ok || rel_ok;
}

SolverReport solve_direct(const EigenMatrix& A, EigenVector& x, const EigenVector& b, const SolverOptions& options)
{
    SolverReport report;
    report.initial_residual_norm = (b.eigen() - A.eigen() * x.eigen()).norm();

    Eigen::SparseMatrix<Real, Eigen::ColMajor, EigenMatrix::StorageIndex> Acol = A.eigen();
    Eigen::SparseLU<decltype(Acol)> solver;
    solver.analyzePattern(Acol);
    solver.factorize(Acol);

    FE_THROW_IF(solver.info() != Eigen::Success, FEException,
                "EigenLinearSolver (direct): factorization failed");

    x.eigen() = solver.solve(b.eigen());
    FE_THROW_IF(solver.info() != Eigen::Success, FEException,
                "EigenLinearSolver (direct): solve failed");

    report.final_residual_norm = (b.eigen() - A.eigen() * x.eigen()).norm();
    const Real denom = std::max<Real>(report.initial_residual_norm, 1e-30);
    report.relative_residual = report.final_residual_norm / denom;
    report.iterations = 1;

    report.converged = check_convergence(report, options);

    report.message = "direct";
    return report;
}

template <typename IterativeSolver>
SolverReport solve_iterative(const EigenMatrix& A,
                             EigenVector& x,
                             const EigenVector& b,
                             IterativeSolver& solver,
                             const SolverOptions& options)
{
    SolverReport report;

    if (!options.use_initial_guess) {
        x.eigen().setZero();
    }

    const auto r0 = (b.eigen() - A.eigen() * x.eigen());
    report.initial_residual_norm = r0.norm();

    solver.setMaxIterations(options.max_iter);
    solver.setTolerance(options.rel_tol);
    solver.compute(A.eigen());

    if (options.use_initial_guess) {
        x.eigen() = solver.solveWithGuess(b.eigen(), x.eigen());
    } else {
        x.eigen() = solver.solve(b.eigen());
    }

    report.iterations = solver.iterations();
    const auto rf = (b.eigen() - A.eigen() * x.eigen());
    report.final_residual_norm = rf.norm();

    const Real denom = std::max<Real>(report.initial_residual_norm, 1e-30);
    report.relative_residual = report.final_residual_norm / denom;

    report.converged = check_convergence(report, options);

    report.message = "iterative";
    return report;
}

} // namespace

SolverReport EigenLinearSolver::solve(const GenericMatrix& A_in,
                                      GenericVector& x_in,
                                      const GenericVector& b_in)
{
    const auto* A = dynamic_cast<const EigenMatrix*>(&A_in);
    auto* x = dynamic_cast<EigenVector*>(&x_in);
    const auto* b = dynamic_cast<const EigenVector*>(&b_in);

    FE_THROW_IF(!A || !x || !b, InvalidArgumentException, "EigenLinearSolver::solve: backend mismatch");
    FE_THROW_IF(options_.preconditioner == PreconditionerType::FieldSplit, NotImplementedException,
                "EigenLinearSolver::solve: field-split preconditioning not supported");
    FE_THROW_IF(A->numRows() != A->numCols(), NotImplementedException,
                "EigenLinearSolver::solve: rectangular systems not implemented");
    FE_THROW_IF(A->numCols() != b->size() || x->size() != b->size(), InvalidArgumentException,
                "EigenLinearSolver::solve: size mismatch");

    switch (options_.method) {
        case SolverMethod::Direct:
            return solve_direct(*A, *x, *b, options_);

        case SolverMethod::CG: {
            FE_THROW_IF(options_.preconditioner == PreconditionerType::AMG, NotImplementedException,
                        "EigenLinearSolver::solve: AMG preconditioning not supported");
            if (options_.preconditioner == PreconditionerType::Diagonal ||
                options_.preconditioner == PreconditionerType::RowColumnScaling) {
                Eigen::ConjugateGradient<EigenMatrix::SparseMat, Eigen::Lower | Eigen::Upper,
                                         Eigen::DiagonalPreconditioner<Real>> solver;
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            Eigen::ConjugateGradient<EigenMatrix::SparseMat, Eigen::Lower | Eigen::Upper,
                                     Eigen::IdentityPreconditioner> solver;
            return solve_iterative(*A, *x, *b, solver, options_);
        }

        case SolverMethod::BiCGSTAB: {
            FE_THROW_IF(options_.preconditioner == PreconditionerType::AMG, NotImplementedException,
                        "EigenLinearSolver::solve: AMG preconditioning not supported");
            if (options_.preconditioner == PreconditionerType::Diagonal ||
                options_.preconditioner == PreconditionerType::RowColumnScaling) {
                Eigen::BiCGSTAB<EigenMatrix::SparseMat, Eigen::DiagonalPreconditioner<Real>> solver;
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            if (options_.preconditioner == PreconditionerType::ILU) {
                Eigen::BiCGSTAB<EigenMatrix::SparseMat, Eigen::IncompleteLUT<Real>> solver;
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            Eigen::BiCGSTAB<EigenMatrix::SparseMat, Eigen::IdentityPreconditioner> solver;
            return solve_iterative(*A, *x, *b, solver, options_);
        }

        case SolverMethod::GMRES:
        case SolverMethod::FGMRES:
        case SolverMethod::BlockSchur: {
            FE_THROW_IF(options_.preconditioner == PreconditionerType::AMG, NotImplementedException,
                        "EigenLinearSolver::solve: AMG preconditioning not supported");

            const int restart = std::max(1, std::min(options_.max_iter, 30));

            if (options_.preconditioner == PreconditionerType::Diagonal ||
                options_.preconditioner == PreconditionerType::RowColumnScaling) {
                Eigen::GMRES<EigenMatrix::SparseMat, Eigen::DiagonalPreconditioner<Real>> solver;
                solver.set_restart(restart);
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            if (options_.preconditioner == PreconditionerType::ILU) {
                Eigen::GMRES<EigenMatrix::SparseMat, Eigen::IncompleteLUT<Real>> solver;
                solver.set_restart(restart);
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            Eigen::GMRES<EigenMatrix::SparseMat, Eigen::IdentityPreconditioner> solver;
            solver.set_restart(restart);
            return solve_iterative(*A, *x, *b, solver, options_);
        }
        default:
            FE_THROW(NotImplementedException, "EigenLinearSolver::solve: unknown solver method");
    }
}

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp
