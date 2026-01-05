/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsLinearSolver.h"

#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Core/FEException.h"

#include "Array.h"
#include "Vector.h"
#include "consts.h"
#include "Backends/FSILS/liner_solver/fsils_api.hpp"
#include "Backends/FSILS/liner_solver/fils_struct.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mpi.h>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

FsilsLinearSolver::FsilsLinearSolver(const SolverOptions& options)
{
    setOptions(options);
}

void FsilsLinearSolver::setOptions(const SolverOptions& options)
{
    FE_THROW_IF(options.max_iter <= 0, InvalidArgumentException, "FsilsLinearSolver: max_iter must be > 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "FsilsLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "FsilsLinearSolver: abs_tol must be >= 0");
    FE_THROW_IF(options.use_initial_guess, NotImplementedException, "FsilsLinearSolver: initial guess not supported");
    options_ = options;
}

namespace {

fsi_linear_solver::LinearSolverType to_fsils_solver(SolverMethod method)
{
    switch (method) {
        case SolverMethod::CG: return fsi_linear_solver::LS_TYPE_CG;
        case SolverMethod::GMRES: return fsi_linear_solver::LS_TYPE_GMRES;
        case SolverMethod::FGMRES: return fsi_linear_solver::LS_TYPE_GMRES;
        case SolverMethod::BiCGSTAB: return fsi_linear_solver::LS_TYPE_BICGS;
        case SolverMethod::BlockSchur: return fsi_linear_solver::LS_TYPE_NS;
        case SolverMethod::Direct:
        default:
            FE_THROW(NotImplementedException, "FsilsLinearSolver: direct solve not supported by FSILS");
    }
}

consts::PreconditionerType to_fsils_prec(const SolverOptions& options)
{
    if (options.fsils_use_rcs || options.preconditioner == PreconditionerType::RowColumnScaling) {
        return consts::PreconditionerType::PREC_RCS;
    }

    switch (options.preconditioner) {
        case PreconditionerType::None:
        case PreconditionerType::Diagonal:
        case PreconditionerType::ILU:
        case PreconditionerType::AMG:
            // FSILS' solve path expects the diagonal/scale work vectors to be initialized by a preconditioner
            // routine (it always applies Wc element-wise after the Krylov solve). Treat unsupported/none as the
            // built-in diagonal preconditioner for correctness.
            return consts::PreconditionerType::PREC_FSILS;
        case PreconditionerType::FieldSplit:
            FE_THROW(NotImplementedException, "FsilsLinearSolver: field-split preconditioning not supported");
        default: return consts::PreconditionerType::PREC_NONE;
    }
}

} // namespace

SolverReport FsilsLinearSolver::solve(const GenericMatrix& A_in,
                                      GenericVector& x_in,
                                      const GenericVector& b_in)
{
    const auto* A = dynamic_cast<const FsilsMatrix*>(&A_in);
    auto* x = dynamic_cast<FsilsVector*>(&x_in);
    const auto* b = dynamic_cast<const FsilsVector*>(&b_in);

    FE_THROW_IF(!A || !x || !b, InvalidArgumentException, "FsilsLinearSolver::solve: backend mismatch");
    FE_THROW_IF(A->numRows() != A->numCols(), NotImplementedException,
                "FsilsLinearSolver::solve: rectangular systems not implemented");
    FE_THROW_IF(b->size() != A->numRows() || x->size() != b->size(), InvalidArgumentException,
                "FsilsLinearSolver::solve: size mismatch");

    auto& lhs = *static_cast<fsi_linear_solver::FSILS_lhsType*>(const_cast<void*>(A->fsilsLhsPtr()));
    const int dof = A->fsilsDof();
    FE_THROW_IF(dof <= 0, FEException, "FsilsLinearSolver::solve: invalid FSILS dof");
    FE_THROW_IF(lhs.nNo <= 0, FEException, "FsilsLinearSolver::solve: invalid FSILS lhs.nNo");

    const GlobalIndex expected_local = static_cast<GlobalIndex>(lhs.nNo) * static_cast<GlobalIndex>(dof);
    FE_THROW_IF(static_cast<GlobalIndex>(x->data().size()) != expected_local ||
                    static_cast<GlobalIndex>(b->data().size()) != expected_local,
                FEException, "FsilsLinearSolver::solve: FSILS vectors must have local size lhs.nNo*dof");

    if (options_.method == SolverMethod::BlockSchur) {
        // FSILS NS solver is implemented for nsd=2 or nsd=3 (dof = nsd+1).
        FE_THROW_IF(dof != 3 && dof != 4, NotImplementedException,
                    "FsilsLinearSolver::solve: BlockSchur requires dof=3 (2D) or dof=4 (3D) for FSILS NS solver");
    }

    // FSILS may modify the matrix values during preconditioning/solve; work on a copy.
    const GlobalIndex nnz = A->fsilsNnz();
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t value_count = static_cast<std::size_t>(nnz) * block_size;
    std::vector<Real> values_work(value_count);
    std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work.data());

    // Copy RHS into solution buffer (FSILS uses Ri as in/out).
    x->data() = b->data();

    Array<double> Ri(dof, lhs.nNo, x->data().data());
    FE_THROW_IF(nnz > static_cast<GlobalIndex>(std::numeric_limits<int>::max()), InvalidArgumentException,
                "FsilsLinearSolver::solve: nnz exceeds FSILS int index range");
    Array<double> Val(dof * dof, static_cast<int>(nnz), values_work.data());

    fsi_linear_solver::FSILS_lsType ls{};
    int effective_max_iter = options_.max_iter;
    if (options_.method == SolverMethod::BlockSchur) {
        // FSILS NS solver uses RI.mItr as a basis dimension and allocates O(nNo * mItr) workspace.
        // Treat very large generic max_iter as "unset" and fall back to the FSILS default (10) for safety.
        const int safe_max_iter = (options_.max_iter > 50) ? 10 : options_.max_iter;
        effective_max_iter = safe_max_iter;
        fsi_linear_solver::fsils_ls_create(ls,
                                           fsi_linear_solver::LS_TYPE_NS,
                                           options_.rel_tol,
                                           options_.abs_tol,
                                           safe_max_iter);
    } else {
        fsi_linear_solver::fsils_ls_create(ls,
                                           to_fsils_solver(options_.method),
                                           options_.rel_tol,
                                           options_.abs_tol,
                                           options_.max_iter);
    }

    Vector<int> incL(0);
    Vector<double> res(0);

    const auto prec = to_fsils_prec(options_);

    // FSILS iterative routines assume overlap contributions have been communicated before
    // norm/dot operations. Apply FSILS COMMU to the working RHS (Ri) in internal ordering.
    {
        std::vector<double> r_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(lhs.nNo), 0.0);
        for (int a = 0; a < lhs.nNo; ++a) {
            const int internal = lhs.map(a);
            for (int c = 0; c < dof; ++c) {
                r_internal[static_cast<std::size_t>(c) +
                           static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)] =
                    x->data()[static_cast<std::size_t>(c) +
                              static_cast<std::size_t>(a) * static_cast<std::size_t>(dof)];
            }
        }

        Array<double> R_int(dof, lhs.nNo, r_internal.data());
        fsi_linear_solver::fsils_commuv(lhs, dof, R_int);

        // Map back to old local ordering expected by fsils_solve.
        for (int a = 0; a < lhs.nNo; ++a) {
            const int internal = lhs.map(a);
            for (int c = 0; c < dof; ++c) {
                x->data()[static_cast<std::size_t>(c) +
                          static_cast<std::size_t>(a) * static_cast<std::size_t>(dof)] =
                    r_internal[static_cast<std::size_t>(c) +
                               static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)];
            }
        }
    }

    SolverReport report;
    report.initial_residual_norm = x->norm();

    fsi_linear_solver::fsils_solve(lhs, ls, dof, Ri, Val, prec, incL, res);

    // Populate diagnostics from FSILS internal report (RI is used across solvers).
    report.iterations = ls.RI.itr;
    report.final_residual_norm = ls.RI.fNorm;
    const Real denom = std::max<Real>(report.initial_residual_norm, 1e-30);
    report.relative_residual = report.final_residual_norm / denom;
    report.converged = ls.RI.suc;
    report.message = "fsils";

    // FSILS does not robustly report breakdowns for singular/ill-posed systems; guard against
    // NaNs/infs and corrupted iteration counts so the FE API remains predictable.
    const auto is_finite = [](Real v) { return std::isfinite(static_cast<double>(v)); };
    bool x_finite = true;
    for (const auto v : x->data()) {
        if (!is_finite(v)) {
            x_finite = false;
            break;
        }
    }
    const bool iters_ok = (report.iterations >= 0) && (report.iterations <= effective_max_iter);
    if (!is_finite(report.final_residual_norm) || !is_finite(report.relative_residual) || !iters_ok || !x_finite) {
        x->zero();
        report.converged = false;
        report.iterations = std::clamp(report.iterations, 0, effective_max_iter);
        report.final_residual_norm = std::numeric_limits<Real>::infinity();
        report.relative_residual = std::numeric_limits<Real>::infinity();
        report.message = "fsils (numerical breakdown)";
    }

    return report;
}

} // namespace backends
} // namespace FE
} // namespace svmp
