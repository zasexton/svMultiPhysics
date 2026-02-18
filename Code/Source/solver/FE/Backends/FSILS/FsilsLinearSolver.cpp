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
#include <exception>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <numeric>
#include <map>
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
    if (options.fsils_ns_gm_max_iter) {
        FE_THROW_IF(*options.fsils_ns_gm_max_iter <= 0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_ns_gm_max_iter must be > 0");
    }
    if (options.fsils_ns_cg_max_iter) {
        FE_THROW_IF(*options.fsils_ns_cg_max_iter <= 0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_ns_cg_max_iter must be > 0");
    }
    if (options.fsils_ns_gm_rel_tol) {
        FE_THROW_IF(*options.fsils_ns_gm_rel_tol < 0.0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_ns_gm_rel_tol must be >= 0");
    }
    if (options.fsils_ns_cg_rel_tol) {
        FE_THROW_IF(*options.fsils_ns_cg_rel_tol < 0.0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_ns_cg_rel_tol must be >= 0");
    }
    options_ = options;
}

void FsilsLinearSolver::setRankOneUpdates(std::span<const RankOneUpdate> updates)
{
    rank_one_updates_.assign(updates.begin(), updates.end());
}

void FsilsLinearSolver::setDirichletDofs(std::span<const GlobalIndex> dofs)
{
    dirichlet_dofs_.assign(dofs.begin(), dofs.end());
    std::sort(dirichlet_dofs_.begin(), dirichlet_dofs_.end());
    dirichlet_dofs_.erase(std::unique(dirichlet_dofs_.begin(), dirichlet_dofs_.end()),
                          dirichlet_dofs_.end());
}

void FsilsLinearSolver::setEffectiveTimeStep(double dt_eff)
{
    if (std::isfinite(dt_eff) && dt_eff > 0.0) {
        dt_eff_ = dt_eff;
    } else {
        dt_eff_ = 1.0;
    }
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

fe_fsi_linear_solver::LinearSolverType to_fsils_solver(SolverMethod method)
{
    switch (method) {
        case SolverMethod::CG: return fe_fsi_linear_solver::LS_TYPE_CG;
        case SolverMethod::GMRES: return fe_fsi_linear_solver::LS_TYPE_GMRES;
        case SolverMethod::FGMRES: return fe_fsi_linear_solver::LS_TYPE_GMRES;
        case SolverMethod::BiCGSTAB: return fe_fsi_linear_solver::LS_TYPE_BICGS;
        case SolverMethod::BlockSchur: return fe_fsi_linear_solver::LS_TYPE_NS;
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

    auto& lhs = *static_cast<fe_fsi_linear_solver::FSILS_lhsType*>(const_cast<void*>(A->fsilsLhsPtr()));
    const int dof = A->fsilsDof();
    const int nsd = dof - 1; // velocity components (FSILS convention: dof = nsd + 1 for NS)
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

    // Copy RHS into solution buffer (FSILS uses Ri as in/out). Avoid reallocation so
    // Array views into `x->data()` remain valid across fallback solves.
    auto& x_data = x->data();
    const auto& b_data = b->data();
    FE_THROW_IF(x_data.size() != b_data.size(), FEException, "FsilsLinearSolver::solve: RHS size mismatch");
    std::copy(b_data.begin(), b_data.end(), x_data.begin());

    Array<double> Ri(dof, lhs.nNo, x_data.data());
    FE_THROW_IF(nnz > static_cast<GlobalIndex>(std::numeric_limits<int>::max()), InvalidArgumentException,
                "FsilsLinearSolver::solve: nnz exceeds FSILS int index range");
    Array<double> Val(dof * dof, static_cast<int>(nnz), values_work.data());

    // Optional scaling used for the Navier-Stokes (BlockSchur) solver path.
    //
    // The legacy solver scales resistance-type coupled BC tangent contributions by (gamma*dt),
    // where gamma is the generalized-α parameter. The OOP solver provides the effective stage
    // dt via LinearSolver::setEffectiveTimeStep().
    //
    // FSILS' NS solver assumes the saddle-point structure G ≈ -D^T. If we only left-scale the
    // momentum rows, that relationship is broken. To preserve it while still improving
    // conditioning, apply:
    //   - left scaling on momentum rows by `stage_scale`
    //   - right scaling on pressure columns by `1/stage_scale`
    // and rescale the solved pressure component back afterward.
    double stage_scale = 1.0;
    if (options_.method == SolverMethod::BlockSchur && (dof == 3 || dof == 4)) {
        if (std::isfinite(dt_eff_) && dt_eff_ > 0.0) {
            stage_scale = dt_eff_;
        }
    }

    auto applyStageScalingToMatrix = [&]() {
        if (stage_scale == 1.0) {
            return;
        }
        const Real s = static_cast<Real>(stage_scale);
        const Real inv_s = static_cast<Real>(1.0 / stage_scale);

        // Left-scale momentum rows (0..nsd-1).
        for (GlobalIndex bi = 0; bi < nnz; ++bi) {
            Real* blk = values_work.data() + static_cast<std::size_t>(bi) * block_size;
            for (int r = 0; r < nsd; ++r) {
                for (int c = 0; c < dof; ++c) {
                    blk[static_cast<std::size_t>(r * dof + c)] *= s;
                }
            }
            // Right-scale pressure column (nsd) to preserve G ≈ -D^T.
            for (int r = 0; r < dof; ++r) {
                blk[static_cast<std::size_t>(r * dof + nsd)] *= inv_s;
            }
        }
    };

    auto restoreAndScaleMatrixValues = [&]() {
        std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work.data());
        applyStageScalingToMatrix();
    };

    applyStageScalingToMatrix();

    // FSILS' NS (BlockSchur) solver assumes a saddle-point structure where the
    // pressure-velocity block is the negative transpose of the velocity-pressure
    // block (up to sparsity transposition). Some FE formulations / linearizations
    // may introduce small inconsistencies that can destabilize the NS solver's
    // Schur complement solve (CG requires an SPD operator).
    //
    // To match legacy solver expectations and improve robustness, enforce:
    //   D(i<-j) = -G(j<-i)^T
    // by overwriting the bottom-left block entries at the transposed sparsity
    // location with the negative of the top-right block entries.
    //
    // This is a no-op when the assembled Jacobian already satisfies the relation.
    if (options_.method == SolverMethod::BlockSchur && (dof == 3 || dof == 4)) {
        const int nNo = lhs.nNo;
        const int nnz_int = lhs.nnz;
        if (nNo > 0 && nnz_int > 0) {
            auto* cols = lhs.colPtr.data();
            const auto find_entry = [&](int row, int col) -> fe_fsi_linear_solver::fsils_int {
                const auto start = lhs.rowPtr(0, row);
                const auto end = lhs.rowPtr(1, row);
                if (start < 0 || end < start) {
                    return -1;
                }
                const auto len = end - start + 1;
                auto* begin = cols + start;
                auto* finish = begin + len;
                const auto it = std::lower_bound(begin, finish, static_cast<fe_fsi_linear_solver::fsils_int>(col));
                if (it == finish || *it != col) {
                    return -1;
                }
                return static_cast<fe_fsi_linear_solver::fsils_int>(it - cols);
            };

            const std::size_t blk_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
            double max_abs_diff = 0.0;
            double max_abs_ref = 0.0;

            for (fe_fsi_linear_solver::fsils_int row = 0; row < nNo; ++row) {
                const auto start = lhs.rowPtr(0, row);
                const auto end = lhs.rowPtr(1, row);
                if (start < 0 || end < start) {
                    continue;
                }
                for (auto idx = start; idx <= end; ++idx) {
                    const auto col = cols[idx];
                    if (col < 0 || col >= nNo) {
                        continue;
                    }

                    const auto idx_t = find_entry(col, row);
                    if (idx_t < 0 || idx_t >= nnz_int) {
                        continue;
                    }

                    Real* blk = values_work.data() + static_cast<std::size_t>(idx) * blk_size;
                    Real* blk_t = values_work.data() + static_cast<std::size_t>(idx_t) * blk_size;

                    for (int c = 0; c < nsd; ++c) {
                        const Real top = blk[static_cast<std::size_t>(c * dof + nsd)];
                        const std::size_t bl_idx = static_cast<std::size_t>(nsd * dof + c);
                        const Real bottom = blk_t[bl_idx];
                        max_abs_ref = std::max(max_abs_ref, static_cast<double>(std::abs(top)));
                        max_abs_ref = std::max(max_abs_ref, static_cast<double>(std::abs(bottom)));
                        const double diff = static_cast<double>(bottom + top);
                        max_abs_diff = std::max(max_abs_diff, std::abs(diff));

                        // Enforce bottom-left at transpose == -top-right at original.
                        blk_t[bl_idx] = -top;
                    }
                }
            }

            if (oopTraceEnabled() && max_abs_ref > 0.0) {
                std::ostringstream oss;
                oss << "FsilsLinearSolver: enforced D=-G^T (max|D+G^T|=" << max_abs_diff
                    << ", rel=" << (max_abs_diff / max_abs_ref) << ")";
                traceLog(oss.str());
            }
        }
    }

	    fe_fsi_linear_solver::FSILS_lsType ls{};
		    if (options_.method == SolverMethod::BlockSchur) {
	        // FSILS NS solver uses RI.mItr as a basis dimension and allocates O(nNo * mItr) workspace.
	        // Treat very large generic max_iter as "unset" and fall back to the FSILS default (10) for safety.
	        const int safe_max_iter = (options_.max_iter > 50) ? 10 : options_.max_iter;
	        if (options_.krylov_dim > 0) {
	            fe_fsi_linear_solver::fsils_ls_create(ls,
	                                               fe_fsi_linear_solver::LS_TYPE_NS,
	                                               options_.rel_tol,
	                                               options_.abs_tol,
	                                               safe_max_iter,
	                                               options_.krylov_dim);
	        } else {
	            fe_fsi_linear_solver::fsils_ls_create(ls,
	                                               fe_fsi_linear_solver::LS_TYPE_NS,
	                                               options_.rel_tol,
	                                               options_.abs_tol,
	                                               safe_max_iter);
	        }

	        // Legacy semantics: GM/CG inherit absTol and Krylov dimension from RI.
	        ls.GM.absTol = ls.RI.absTol;
	        ls.CG.absTol = ls.RI.absTol;
	        ls.GM.sD = ls.RI.sD;

	        if (options_.fsils_ns_gm_max_iter) {
	            ls.GM.mItr = *options_.fsils_ns_gm_max_iter;
	        }
	        if (options_.fsils_ns_cg_max_iter) {
	            ls.CG.mItr = *options_.fsils_ns_cg_max_iter;
	        }
	        if (options_.fsils_ns_gm_rel_tol) {
	            ls.GM.relTol = *options_.fsils_ns_gm_rel_tol;
	        }
	        if (options_.fsils_ns_cg_rel_tol) {
	            ls.CG.relTol = *options_.fsils_ns_cg_rel_tol;
	        }
		    } else {
		        const auto method = to_fsils_solver(options_.method);
		        if (method == fe_fsi_linear_solver::LS_TYPE_GMRES) {
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

		            fe_fsi_linear_solver::fsils_ls_create(ls,
		                                               method,
		                                               options_.rel_tol,
		                                               options_.abs_tol,
		                                               mItr,
		                                               sD);
		        } else {
		        fe_fsi_linear_solver::fsils_ls_create(ls,
		                                           method,
		                                           options_.rel_tol,
	                                           options_.abs_tol,
	                                           options_.max_iter);
	        }
	    }

    // Set up FSILS faces from:
    //  - Dirichlet constraints (legacy-equivalent FSILS preconditioner handling)
    //  - rank-1 updates (coupled BC Sherman-Morrison correction)
    const int original_nFaces = lhs.nFaces;
    const int num_dirichlet_faces = (!dirichlet_dofs_.empty() ? 1 : 0);
    const int num_rank_one = static_cast<int>(rank_one_updates_.size());
    const int num_added_faces = num_dirichlet_faces + num_rank_one;

    int dirichlet_face_index = -1;
    int rank_one_face_start = -1;

    auto sort_face_by_glob = [&](fe_fsi_linear_solver::FSILS_faceType& face, int face_dof) {
        if (face.nNo <= 1) {
            return;
        }
        const int face_nNo = face.nNo;
        std::vector<int> perm(static_cast<std::size_t>(face_nNo));
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](int i, int j) { return face.glob(i) < face.glob(j); });

        Vector<int> sorted_glob(face_nNo);
        Array<double> sorted_val(face_dof, face_nNo);
        for (int i = 0; i < face_nNo; ++i) {
            const int src = perm[static_cast<std::size_t>(i)];
            sorted_glob(i) = face.glob(src);
            for (int c = 0; c < face_dof; ++c) {
                sorted_val(c, i) = face.val(c, src);
            }
        }
        face.glob = sorted_glob;
        face.val = sorted_val;
    };

    auto sync_face_val_if_shared = [&](fe_fsi_linear_solver::FSILS_faceType& face, int face_dof) {
        if (lhs.commu.nTasks <= 1) {
            return;
        }

        const int local_has = (face.nNo > 0) ? 1 : 0;
        int total_has = 0;
        MPI_Allreduce(&local_has, &total_has, 1, MPI_INT, MPI_SUM, lhs.commu.comm);

        if (total_has > 1) {
            face.sharedFlag = true;
            Array<double> v(face_dof, lhs.nNo);
            v = 0.0;

            for (int a = 0; a < face.nNo; ++a) {
                const int Ac = face.glob(a);
                for (int i = 0; i < face_dof; ++i) {
                    v(i, Ac) = face.val(i, a);
                }
            }

            fe_fsi_linear_solver::fsils_commuv(lhs, face_dof, v);

            for (int a = 0; a < face.nNo; ++a) {
                const int Ac = face.glob(a);
                for (int i = 0; i < face_dof; ++i) {
                    face.val(i, a) = v(i, Ac);
                }
            }
        }
    };

    if (num_added_faces > 0 && dof > 0) {
        const auto shared = A->shared();
        FE_CHECK_NOT_NULL(shared.get(), "FsilsLinearSolver: FsilsShared for face setup");

        const int new_nFaces = original_nFaces + num_added_faces;
        lhs.face.resize(static_cast<std::size_t>(new_nFaces));
        lhs.nFaces = new_nFaces;

        int next_face = original_nFaces;

        if (num_dirichlet_faces > 0) {
            dirichlet_face_index = next_face++;

            // Node mask: old_local_node -> per-component 0/1 mask (0 for Dirichlet components).
            std::map<int, std::vector<double>> node_mask;
            for (const auto dof_idx : dirichlet_dofs_) {
                GlobalIndex fsils_dof = dof_idx;
                if (shared->dof_permutation) {
                    const auto idx = static_cast<std::size_t>(dof_idx);
                    if (idx < shared->dof_permutation->forward.size()) {
                        fsils_dof = shared->dof_permutation->forward[idx];
                    }
                }
                if (fsils_dof < 0) {
                    continue;
                }

                const int node = static_cast<int>(fsils_dof / dof);
                const int comp = static_cast<int>(fsils_dof % dof);
                if (comp < 0 || comp >= dof) {
                    continue;
                }

                const int old_local = shared->globalNodeToOld(node);
                if (old_local < 0 || old_local >= lhs.nNo) {
                    continue;
                }

                auto& mask = node_mask[old_local];
                if (mask.empty()) {
                    mask.assign(static_cast<std::size_t>(dof), 1.0);
                }
                mask[static_cast<std::size_t>(comp)] = 0.0;
            }

            auto& face = lhs.face[static_cast<std::size_t>(dirichlet_face_index)];
            const int face_nNo = static_cast<int>(node_mask.size());
            face.nNo = face_nNo;
            face.dof = dof;
            face.bGrp = fe_fsi_linear_solver::BcType::BC_TYPE_Dir;

	            if (face_nNo > 0) {
	                face.glob.resize(face_nNo);
	                face.val.resize(dof, face_nNo);
	                face.valM.resize(dof, face_nNo);
	                face.val = 1.0;
	                face.valM = 0.0;

                int a = 0;
                for (const auto& [old_local, mask] : node_mask) {
                    face.glob(a) = lhs.map(old_local);
                    for (int c = 0; c < dof; ++c) {
                        face.val(c, a) = mask[static_cast<std::size_t>(c)];
                    }
                    ++a;
                }
	
	                sort_face_by_glob(face, dof);
	            }
	            // Must be called collectively across ranks (uses MPI_Allreduce / COMMU).
	            sync_face_val_if_shared(face, dof);
	
	            face.foC = true;
	            face.coupledFlag = false;
	            face.incFlag = true;

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "FsilsLinearSolver: Dirichlet face " << dirichlet_face_index
                    << " nNo=" << face_nNo
                    << " dirichlet_dofs=" << dirichlet_dofs_.size();
                traceLog(oss.str());
            }
        }

        rank_one_face_start = next_face;
        for (int u = 0; u < num_rank_one; ++u) {
            const auto& upd = rank_one_updates_[static_cast<std::size_t>(u)];
            const int faIn = rank_one_face_start + u;

            // Group sparse DOF entries by node, keeping only velocity components.
            // Map: old_local_node -> [component -> value]
            std::map<int, std::map<int, Real>> node_data;
            for (const auto& [dof_idx, val] : upd.v) {
                GlobalIndex fsils_dof = dof_idx;
                if (shared->dof_permutation) {
                    const auto idx = static_cast<std::size_t>(dof_idx);
                    if (idx < shared->dof_permutation->forward.size()) {
                        fsils_dof = shared->dof_permutation->forward[idx];
                    }
                }

                // Skip unmapped DOFs (permutation returns -1 for DOFs not present on this rank).
                if (fsils_dof < 0) {
                    continue;
                }

                const int node = static_cast<int>(fsils_dof / dof);
                const int comp = static_cast<int>(fsils_dof % dof);

                // Only velocity components (0..nsd-1), skip pressure (nsd).
                if (comp >= nsd) {
                    continue;
                }

                const int old_local = shared->globalNodeToOld(node);
                if (old_local < 0 || old_local >= lhs.nNo) {
                    continue;
                }
                node_data[old_local][comp] = val;
            }

            const int face_nNo = static_cast<int>(node_data.size());

            // Set up face data directly to avoid Vector/Array zero-size constructor issues.
            // fsils_bc_create does MPI communication for shared faces.
            {
                auto& face = lhs.face[static_cast<std::size_t>(faIn)];
                face.nNo = face_nNo;
                face.dof = nsd;
                face.bGrp = fe_fsi_linear_solver::BcType::BC_TYPE_Neu;

	                if (face_nNo > 0) {
	                    face.glob.resize(face_nNo);
	                    face.val.resize(nsd, face_nNo);
	                    face.valM.resize(nsd, face_nNo);
	                    face.val = 0.0;
	                    face.valM = 0.0;

                    int a = 0;
                    for (const auto& [old_local, comp_vals] : node_data) {
                        face.glob(a) = lhs.map(old_local);
                        for (const auto& [comp, val] : comp_vals) {
                            face.val(comp, a) = static_cast<double>(val);
                        }
                        ++a;
                    }
	
	                    sort_face_by_glob(face, nsd);
	                }
	                // Must be called collectively across ranks (uses MPI_Allreduce / COMMU).
	                sync_face_val_if_shared(face, nsd);
	                // else: face_nNo=0, glob/val/valM stay default-constructed (empty)
	
	                face.foC = true;
	                face.coupledFlag = true;
                face.incFlag = true;
            }

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "FsilsLinearSolver: rank-1 update " << u
                    << " -> FSILS face " << faIn
                    << " nNo=" << face_nNo
                    << " sigma=" << static_cast<double>(upd.sigma)
                    << " v_entries=" << upd.v.size();
                traceLog(oss.str());
            }
        }
    }

    // Build incL and res vectors for face activation.
    // When no faces exist, pass empty vectors (original behavior).
    // Note: must use default constructors, not Vector(0), because Vector(0) leaves
    // data_ uninitialized (legacy Fortran compat), causing crashes in resize().
    Vector<int> incL;
    Vector<double> res;
    if (lhs.nFaces > 0) {
        const int total_faces = lhs.nFaces;
        incL.resize(total_faces);
        res.resize(total_faces);
        for (int f = 0; f < total_faces; ++f) {
            incL(f) = 1;
            res(f) = 0.0;
        }
        if (num_rank_one > 0 && rank_one_face_start >= 0) {
            // Set resistance values for rank-1 faces.
            for (int u = 0; u < num_rank_one; ++u) {
                const int faIn = rank_one_face_start + u;
                res(faIn) = static_cast<double>(rank_one_updates_[static_cast<std::size_t>(u)].sigma) * stage_scale;
            }
        }
    }

    const auto prec = to_fsils_prec(options_);

    auto prepareRhsForFsils = [&]() {
        // FSILS iterative routines assume overlap contributions have been communicated before
        // norm/dot operations. Apply FSILS COMMU to the working RHS (Ri) in internal ordering.
        std::vector<double> r_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(lhs.nNo), 0.0);
        for (int a = 0; a < lhs.nNo; ++a) {
            const int internal = lhs.map(a);
            for (int c = 0; c < dof; ++c) {
                r_internal[static_cast<std::size_t>(c) +
                           static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)] =
                    x_data[static_cast<std::size_t>(c) +
                           static_cast<std::size_t>(a) * static_cast<std::size_t>(dof)];
            }
        }

        Array<double> R_int(dof, lhs.nNo, r_internal.data());
        fe_fsi_linear_solver::fsils_commuv(lhs, dof, R_int);

        // Map back to old local ordering expected by fsils_solve.
        for (int a = 0; a < lhs.nNo; ++a) {
            const int internal = lhs.map(a);
            for (int c = 0; c < dof; ++c) {
                x_data[static_cast<std::size_t>(c) +
                       static_cast<std::size_t>(a) * static_cast<std::size_t>(dof)] =
                    r_internal[static_cast<std::size_t>(c) +
                               static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)];
            }
        }

        if (stage_scale != 1.0) {
            const Real s = static_cast<Real>(stage_scale);
            for (int a = 0; a < lhs.nNo; ++a) {
                for (int c = 0; c < nsd; ++c) {
                    Ri(c, a) *= s;
                }
            }
        }
    };

    prepareRhsForFsils();

    SolverReport report;
    report.initial_residual_norm = x->norm();

    bool ns_threw = false;
    std::string ns_error;
    try {
        fe_fsi_linear_solver::fsils_solve(lhs, ls, dof, Ri, Val, prec, incL, res);
    } catch (const std::exception& e) {
        ns_threw = true;
        ns_error = e.what();
    } catch (...) {
        ns_threw = true;
        ns_error = "unknown exception";
    }

    bool used_fallback_gmres = false;
    if (options_.method == SolverMethod::BlockSchur) {
        // FSILS solves are collective. Ensure fallback decisions are consistent across ranks:
        // if any rank throws or reports non-convergence, all ranks must fall back.
        int local_fail = (ns_threw || !ls.RI.suc) ? 1 : 0;
        int local_threw = ns_threw ? 1 : 0;
        int local_not_suc = (!ns_threw && !ls.RI.suc) ? 1 : 0;

        int any_fail = local_fail;
        int any_threw = local_threw;
        int any_not_suc = local_not_suc;
        if (lhs.commu.nTasks > 1) {
            MPI_Allreduce(&local_fail, &any_fail, 1, MPI_INT, MPI_LOR, lhs.commu.comm);
            MPI_Allreduce(&local_threw, &any_threw, 1, MPI_INT, MPI_LOR, lhs.commu.comm);
            MPI_Allreduce(&local_not_suc, &any_not_suc, 1, MPI_INT, MPI_LOR, lhs.commu.comm);
        }

        const bool need_fallback = (any_fail != 0);

        if (need_fallback) {
        used_fallback_gmres = true;
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            if (any_threw != 0) {
                if (ns_threw) {
                    oss << "FsilsLinearSolver::solve: NS solver threw ('" << ns_error << "'); falling back to GMRES.";
                } else {
                    oss << "FsilsLinearSolver::solve: NS solver threw on another rank; falling back to GMRES.";
                }
            } else if (any_not_suc != 0) {
                oss << "FsilsLinearSolver::solve: NS solver did not converge (itr=" << ls.RI.itr
                    << ", fNorm=" << ls.RI.fNorm << "); falling back to GMRES.";
            } else {
                oss << "FsilsLinearSolver::solve: NS solver failed; falling back to GMRES.";
            }
            traceLog(oss.str());
        }

        // Restore pristine matrix/RHS and retry with full-system GMRES.
        restoreAndScaleMatrixValues();
        std::copy(b_data.begin(), b_data.end(), x_data.begin());
        prepareRhsForFsils();

        const int sD = (options_.krylov_dim > 0) ? options_.krylov_dim : 250;
        const int mItr = std::max(1, std::min(5, options_.max_iter)); // limit restarts for safety
        fe_fsi_linear_solver::fsils_ls_create(ls,
                                           fe_fsi_linear_solver::LS_TYPE_GMRES,
                                           options_.rel_tol,
                                           options_.abs_tol,
                                           mItr,
                                           sD);
        fe_fsi_linear_solver::fsils_solve(lhs, ls, dof, Ri, Val, prec, incL, res);
        }
    } else if (ns_threw) {
        // Propagate non-BlockSchur failures (no fallback policy).
        FE_THROW(FEException, "FsilsLinearSolver::solve: FSILS solve threw: " + ns_error);
    }

    if (stage_scale != 1.0) {
        const Real inv_s = static_cast<Real>(1.0 / stage_scale);
        for (int a = 0; a < lhs.nNo; ++a) {
            // Undo right-scaling of the pressure unknown.
            Ri(nsd, a) *= inv_s;
        }
    }

    // Clean up temporary face objects: explicitly free their arrays via clear(),
    // then pop them from the vector. The clear() calls ensure data_ = nullptr
    // so the FSILS_faceType destructor (called by vector::pop_back) won't
    // try to delete[] stale pointers.
    if (num_added_faces > 0) {
        for (int faIn = lhs.nFaces - 1; faIn >= original_nFaces; --faIn) {
            auto& face = lhs.face[static_cast<std::size_t>(faIn)];
            face.glob.clear();
            face.val.clear();
            face.valM.clear();
            face.foC = false;
            face.coupledFlag = false;
            face.incFlag = false;
            face.sharedFlag = false;
            face.nNo = 0;
            face.dof = 0;
            face.nS = 0.0;
            face.res = 0.0;
        }
        lhs.face.resize(static_cast<std::size_t>(original_nFaces));
        lhs.nFaces = original_nFaces;
    }

		    // Populate diagnostics from FSILS internal report (RI is used across solvers).
		    report.iterations = ls.RI.itr;
		    report.final_residual_norm = ls.RI.fNorm;
		    const Real denom = std::max<Real>(report.initial_residual_norm, 1e-30);
		    report.relative_residual = report.final_residual_norm / denom;
		    report.converged = ls.RI.suc;
		    report.message = "fsils";
		    if (used_fallback_gmres) {
		        report.message = "fsils (fallback gmres)";
		    }

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
			    if (ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_GMRES) {
			        // FSILS GMRES counts iterations as `mItr * (sD + 1)` where:
			        // - RI.mItr: restart count (outer)
			        // - RI.sD:   Krylov subspace dimension (inner)
			        //
			        // For BlockSchur solves, we may fall back to GMRES with a different (mItr, sD)
			        // than the user-provided BlockSchur `max_iter`. Use the LS settings to derive
			        // the theoretical maximum iteration count.
			        const long long mItr = static_cast<long long>(std::max(1, ls.RI.mItr));
			        const long long sD = static_cast<long long>(std::max(0, ls.RI.sD));
			        const long long expected = mItr * (sD + 1LL);
			        if (expected > 0 && expected < static_cast<long long>(std::numeric_limits<int>::max())) {
			            max_expected_iters = static_cast<int>(expected);
			        } else {
			            max_expected_iters = std::numeric_limits<int>::max();
			        }
			    } else if (ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_CG ||
			               ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_BICGS ||
			               ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_NS) {
			        // For these FSILS solvers, RI.mItr is the primary iteration cap.
			        max_expected_iters = std::max(0, ls.RI.mItr);
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
		    } else {
		        // FSILS occasionally reports suc=false even when the FE convergence criteria are met,
		        // especially for nearly-zero RHS where relative residual is ill-conditioned. Apply the
		        // FE criteria as a post-check to avoid spurious Newton failures.
		        const Real rel_tol = std::max<Real>(options_.rel_tol, 0.0);
		        const Real abs_tol = std::max<Real>(options_.abs_tol, 0.0);
		        const Real target = std::max(abs_tol, rel_tol * report.initial_residual_norm);
		        if (!report.converged && report.final_residual_norm <= target) {
		            report.converged = true;
		            report.message = used_fallback_gmres ? "fsils (fallback gmres)" : "fsils";
		        } else if (!report.converged) {
		            report.message = "fsils (not converged; itr=" + std::to_string(report.iterations) +
		                             ", rel=" + std::to_string(report.relative_residual) + ")";
		            if (used_fallback_gmres) {
		                report.message = "fsils (fallback gmres; not converged; itr=" + std::to_string(report.iterations) +
		                                 ", rel=" + std::to_string(report.relative_residual) + ")";
		            }
		        }
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
