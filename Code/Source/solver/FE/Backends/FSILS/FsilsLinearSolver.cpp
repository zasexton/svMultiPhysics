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
#include "Core/Logger.h"

#include "Array.h"
#include "Vector.h"
#include "consts.h"
#include "Backends/FSILS/liner_solver/fsils_api.hpp"
#include "Backends/FSILS/liner_solver/fils_struct.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <string>
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
    FE_THROW_IF(options.krylov_dim < 0, InvalidArgumentException, "FsilsLinearSolver: krylov_dim must be >= 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "FsilsLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "FsilsLinearSolver: abs_tol must be >= 0");
    FE_THROW_IF(options.use_initial_guess, NotImplementedException, "FsilsLinearSolver: initial guess not supported");
    options_ = options;
}

namespace {

[[nodiscard]] bool oopTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

void traceLog(const std::string& msg)
{
    if (!oopTraceEnabled()) {
        return;
    }
    FE_LOG_INFO(msg);
}

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

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: n=" << A->numRows()
            << " dof=" << dof << " nNo=" << lhs.nNo
            << " method=" << solverMethodToString(options_.method)
            << " prec=" << preconditionerToString(options_.preconditioner)
            << " rel_tol=" << options_.rel_tol
            << " abs_tol=" << options_.abs_tol
            << " max_iter=" << options_.max_iter
            << " krylov_dim=" << options_.krylov_dim
            << " fsils_use_rcs=" << (options_.fsils_use_rcs ? 1 : 0);
        traceLog(oss.str());
    }

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
	    if (options_.method == SolverMethod::BlockSchur) {
	        // FSILS NS solver uses RI.mItr as a basis dimension and allocates O(nNo * mItr) workspace.
	        // Treat very large generic max_iter as "unset" and fall back to the FSILS default (10) for safety.
	        const int safe_max_iter = (options_.max_iter > 50) ? 10 : options_.max_iter;
	        fsi_linear_solver::fsils_ls_create(ls,
	                                           fsi_linear_solver::LS_TYPE_NS,
	                                           options_.rel_tol,
	                                           options_.abs_tol,
	                                           safe_max_iter);
		    } else {
		        const auto method = to_fsils_solver(options_.method);
		        if (method == fsi_linear_solver::LS_TYPE_GMRES) {
		            // FSILS GMRES counts iterations as `mItr * (sD + 1)` where:
		            // - RI.mItr: restart count (outer)
		            // - RI.sD:   Krylov subspace dimension (inner)
		            //
		            // FE SolverOptions::max_iter is interpreted as the maximum total Krylov steps, so choose
		            // (mItr, sD) such that the worst-case iteration count does not exceed max_iter.
		            int sD = options_.krylov_dim;
		            if (sD <= 0) {
		                // Keep default restart length modest to limit workspace.
		                sD = std::max(0, std::min(50, options_.max_iter) - 1);
		            }
		            const int sD_max = std::max(0, options_.max_iter - 1);
		            sD = std::clamp(sD, 0, sD_max);

		            const int per_restart = sD + 1;
		            const int mItr = std::max(1, options_.max_iter / std::max(1, per_restart));

		            fsi_linear_solver::fsils_ls_create(ls,
		                                               method,
		                                               options_.rel_tol,
		                                               options_.abs_tol,
		                                               mItr,
		                                               sD);
		        } else {
		        fsi_linear_solver::fsils_ls_create(ls,
		                                           method,
		                                           options_.rel_tol,
	                                           options_.abs_tol,
	                                           options_.max_iter);
	        }
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
			    const auto raw_iterations = report.iterations;
			    const auto raw_fnorm = report.final_residual_norm;
			    const auto raw_rel = report.relative_residual;
		    bool x_finite = true;
		    for (const auto v : x->data()) {
		        if (!is_finite(v)) {
		            x_finite = false;
		            break;
		        }
		    }
			    int max_expected_iters = options_.max_iter;
			    if (options_.method == SolverMethod::GMRES || options_.method == SolverMethod::FGMRES) {
			        // For FSILS GMRES, SolverOptions::max_iter maps to RI.mItr (restart count) while RI.itr
			        // counts total iterations (including inner Krylov steps). Guard against truly-corrupted
			        // counts by comparing against the theoretical maximum iterations.
			        const long long mItr = static_cast<long long>(std::max(1, ls.RI.mItr));
			        const long long sD = static_cast<long long>(std::max(0, ls.RI.sD));
			        const long long expected = mItr * (sD + 1LL);
			        if (expected > 0 && expected < static_cast<long long>(std::numeric_limits<int>::max())) {
			            max_expected_iters = static_cast<int>(expected);
			        } else {
			            max_expected_iters = std::numeric_limits<int>::max();
			        }
			    }
			    const bool iters_ok = (raw_iterations >= 0 && raw_iterations <= max_expected_iters);
			    const bool fnorm_ok = is_finite(raw_fnorm);
			    const bool rel_ok = is_finite(raw_rel);
			    if (!iters_ok || !fnorm_ok || !rel_ok || !x_finite) {
			        report.iterations = std::max(0, std::min(raw_iterations, max_expected_iters));
			        std::string reason;
			        if (!iters_ok) reason += "itr";
			        if (!x_finite) {
		            if (!reason.empty()) reason += ",";
		            reason += "x";
		        }
	        if (!fnorm_ok) {
	            if (!reason.empty()) reason += ",";
	            reason += "fNorm";
	        }
	        if (!rel_ok) {
	            if (!reason.empty()) reason += ",";
	            reason += "rel";
	        }
	        if (reason.empty()) reason = "unknown";

	        x->zero();
	        report.converged = false;
	        report.final_residual_norm = std::numeric_limits<Real>::infinity();
	        report.relative_residual = std::numeric_limits<Real>::infinity();
	        report.message = "fsils (breakdown:" + reason + ")";
		    } else if (!report.converged) {
		        report.message = "fsils (not converged; itr=" + std::to_string(report.iterations) +
		                         ", rel=" + std::to_string(report.relative_residual) + ")";
		    }

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: converged=" << (report.converged ? 1 : 0)
            << " iters=" << report.iterations
            << " r0=" << report.initial_residual_norm
            << " rn=" << report.final_residual_norm
            << " rel=" << report.relative_residual
            << " msg='" << report.message << "'";
        traceLog(oss.str());
    }

    return report;
}

} // namespace backends
} // namespace FE
} // namespace svmp
