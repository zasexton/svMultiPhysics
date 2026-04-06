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
#include "Backends/FSILS/liner_solver/add_bc_mul.h"
#include "Backends/FSILS/liner_solver/fsils_api.hpp"
#include "Backends/FSILS/liner_solver/fils_struct.hpp"
#include "Backends/FSILS/liner_solver/spar_mul.h"
#include <algorithm>
#include <cctype>
#include <exception>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <numeric>
#include <map>
#include <cstring>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

namespace {

struct FsilsResidualCheckResult {
    bool ok{true};
    Real rhs_norm{0.0};
    Real residual_norm{0.0};
    Real relative_residual{0.0};
    std::string detail{};
};

struct FsilsConstraintMeanStats {
    bool valid{false};
    std::uint64_t count{0};
    Real mean{0.0};
    Real rms{0.0};
    Real fluctuation_rms{0.0};
};

enum class FsilsResidualValidationSyncMode : std::uint8_t {
    UpdateGhosts,
    AccumulateOverlap,
    AccumulateThenUpdateGhosts,
};

[[nodiscard]] FsilsResidualValidationSyncMode fsilsResidualValidationSyncMode() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_RESIDUAL_VALIDATION_SYNC");
    if (env == nullptr) {
        return FsilsResidualValidationSyncMode::UpdateGhosts;
    }

    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (value == "accumulate" || value == "accumulateoverlap") {
        return FsilsResidualValidationSyncMode::AccumulateOverlap;
    }
    if (value == "both" || value == "accumulate_then_update" ||
        value == "accumulate-then-update") {
        return FsilsResidualValidationSyncMode::AccumulateThenUpdateGhosts;
    }
    return FsilsResidualValidationSyncMode::UpdateGhosts;
}

[[nodiscard]] bool fsilsCompareFaceOperatorEnabled() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_COMPARE_FACE_OPERATOR");
    if (env == nullptr) {
        return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    return *env != '\0' && *env != '0';
}

void addRankOneUpdatesToProduct(std::span<const RankOneUpdate> updates,
                                FsilsVector& x,
                                FsilsVector& y,
                                fe_fsi_linear_solver::FSILS_commuType& commu)
{
    if (updates.empty()) {
        return;
    }

    auto x_view = x.createAssemblyView();
    auto y_view = y.createAssemblyView();
    FE_CHECK_NOT_NULL(x_view.get(), "FsilsLinearSolver: rank-one x view");
    FE_CHECK_NOT_NULL(y_view.get(), "FsilsLinearSolver: rank-one y view");

    std::vector<Real> dots(updates.size(), Real(0.0));
    for (std::size_t u = 0; u < updates.size(); ++u) {
        Real local_dot = Real(0.0);
        for (const auto& [dof, val] : updates[u].v) {
            local_dot += val * x_view->getVectorEntry(dof);
        }
        dots[u] = local_dot;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && commu.nTasks > 1) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        fe_fsi_linear_solver::fsils_allreduce_sum(dots.data(),
                                                  global_dots.data(),
                                                  static_cast<int>(dots.size()),
                                                  MPI_DOUBLE,
                                                  commu);
        dots.swap(global_dots);
    }
#else
    (void)commu;
#endif

    y_view->beginAssemblyPhase();
    for (std::size_t u = 0; u < updates.size(); ++u) {
        const Real scale = updates[u].sigma * dots[u];
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : updates[u].v) {
            y_view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    y_view->finalizeAssembly();
}

void addReducedFieldUpdatesToProduct(std::span<const ReducedFieldUpdate> updates,
                                     FsilsVector& x,
                                     FsilsVector& y,
                                     fe_fsi_linear_solver::FSILS_commuType& commu)
{
    if (updates.empty()) {
        return;
    }

    auto x_view = x.createAssemblyView();
    auto y_view = y.createAssemblyView();
    FE_CHECK_NOT_NULL(x_view.get(), "FsilsLinearSolver: reduced-update x view");
    FE_CHECK_NOT_NULL(y_view.get(), "FsilsLinearSolver: reduced-update y view");

    std::vector<Real> dots(updates.size(), Real(0.0));
    for (std::size_t u = 0; u < updates.size(); ++u) {
        Real local_dot = Real(0.0);
        for (const auto& [dof, val] : updates[u].right) {
            local_dot += val * x_view->getVectorEntry(dof);
        }
        dots[u] = local_dot;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && commu.nTasks > 1) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        fe_fsi_linear_solver::fsils_allreduce_sum(dots.data(),
                                                  global_dots.data(),
                                                  static_cast<int>(dots.size()),
                                                  MPI_DOUBLE,
                                                  commu);
        dots.swap(global_dots);
    }
#else
    (void)commu;
#endif

    y_view->beginAssemblyPhase();
    for (std::size_t u = 0; u < updates.size(); ++u) {
        const Real scale = updates[u].sigma * dots[u];
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : updates[u].left) {
            y_view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    y_view->finalizeAssembly();
}

void copyVectorOldToInternal(const FsilsVector& src, std::span<Real> dst_internal)
{
    const auto* shared = src.shared();
    FE_CHECK_NOT_NULL(shared, "FsilsLinearSolver: shared layout for old->internal vector copy");

    const int dof = shared->dof;
    const int nNo = shared->lhs.nNo;
    const auto expected_size =
        static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo);
    FE_THROW_IF(dst_internal.size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: old->internal vector copy size mismatch");
    FE_THROW_IF(src.data().size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: old->internal source size mismatch");

    const auto& lhs = shared->lhs;
    const auto& src_data = src.data();
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        const std::size_t src_base =
            static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
        const std::size_t dst_base =
            static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
        for (int c = 0; c < dof; ++c) {
            dst_internal[dst_base + static_cast<std::size_t>(c)] =
                src_data[src_base + static_cast<std::size_t>(c)];
        }
    }
}

void copyVectorInternalToOld(std::span<const Real> src_internal, FsilsVector& dst)
{
    const auto* shared = dst.shared();
    FE_CHECK_NOT_NULL(shared, "FsilsLinearSolver: shared layout for internal->old vector copy");

    const int dof = shared->dof;
    const int nNo = shared->lhs.nNo;
    const auto expected_size =
        static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo);
    FE_THROW_IF(src_internal.size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: internal->old vector copy size mismatch");
    FE_THROW_IF(dst.data().size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: internal->old destination size mismatch");
    auto& dst_data = dst.data();

    if (!shared->old_of_internal.empty()) {
        FE_THROW_IF(static_cast<int>(shared->old_of_internal.size()) != nNo,
                    FEException,
                    "FsilsLinearSolver: invalid old_of_internal size");
        for (int internal = 0; internal < nNo; ++internal) {
            const int old = shared->old_of_internal[static_cast<std::size_t>(internal)];
            FE_THROW_IF(old < 0 || old >= nNo,
                        FEException,
                        "FsilsLinearSolver: invalid old_of_internal entry");
            const std::size_t src_base =
                static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            const std::size_t dst_base =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                dst_data[dst_base + static_cast<std::size_t>(c)] =
                    src_internal[src_base + static_cast<std::size_t>(c)];
            }
        }
        return;
    }

    const auto& lhs = shared->lhs;
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        const std::size_t src_base =
            static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
        const std::size_t dst_base =
            static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
        for (int c = 0; c < dof; ++c) {
            dst_data[dst_base + static_cast<std::size_t>(c)] =
                src_internal[src_base + static_cast<std::size_t>(c)];
        }
    }
}

} // namespace

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
    if (options.fsils_blockschur_gm_max_iter) {
        FE_THROW_IF(*options.fsils_blockschur_gm_max_iter <= 0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_gm_max_iter must be > 0");
    }
    if (options.fsils_blockschur_cg_max_iter) {
        FE_THROW_IF(*options.fsils_blockschur_cg_max_iter <= 0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_cg_max_iter must be > 0");
    }
    if (options.fsils_blockschur_gm_rel_tol) {
        FE_THROW_IF(*options.fsils_blockschur_gm_rel_tol < 0.0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_gm_rel_tol must be >= 0");
    }
    if (options.fsils_blockschur_cg_rel_tol) {
        FE_THROW_IF(*options.fsils_blockschur_cg_rel_tol < 0.0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_cg_rel_tol must be >= 0");
    }
    options_ = options;
}

void FsilsLinearSolver::setRankOneUpdates(std::span<const RankOneUpdate> updates)
{
    // Only dirty face cache if the update set actually changed.
    // When both old and new are empty (common case), skip the dirty flag.
    const bool was_empty = rank_one_updates_.empty();
    const bool now_empty = updates.empty();
    rank_one_updates_.assign(updates.begin(), updates.end());
    if (was_empty && now_empty) {
        return;  // No change — don't dirty face cache.
    }
    faces_dirty_ = true;
}

void FsilsLinearSolver::setReducedFieldUpdates(std::span<const ReducedFieldUpdate> updates)
{
    reduced_field_updates_.assign(updates.begin(), updates.end());
}

void FsilsLinearSolver::setGroupedBorderedFieldCouplings(
    std::span<const GroupedBorderedFieldCoupling> groups)
{
    grouped_bordered_field_couplings_.assign(groups.begin(), groups.end());
}

void FsilsLinearSolver::setDirichletDofs(std::span<const GlobalIndex> dofs)
{
    std::vector<GlobalIndex> new_dofs(dofs.begin(), dofs.end());
    std::sort(new_dofs.begin(), new_dofs.end());
    new_dofs.erase(std::unique(new_dofs.begin(), new_dofs.end()), new_dofs.end());
    if (new_dofs != dirichlet_dofs_) {
        dirichlet_dofs_ = std::move(new_dofs);
        faces_dirty_ = true;
    }
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
        case SolverMethod::PGMRES: return fe_fsi_linear_solver::LS_TYPE_GMRES;
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

fe_fsi_linear_solver::SchurPreconditionerType
to_fsils_blockschur_preconditioner(FsilsBlockSchurSchurPreconditioner pc)
{
    using fe_fsi_linear_solver::SchurPreconditionerType;
    switch (pc) {
        case FsilsBlockSchurSchurPreconditioner::DiagL: return SchurPreconditionerType::DIAG_L;
        case FsilsBlockSchurSchurPreconditioner::BlockDiagL: return SchurPreconditionerType::BLOCKDIAG_L;
        case FsilsBlockSchurSchurPreconditioner::ILUL: return SchurPreconditionerType::ILU_L;
        case FsilsBlockSchurSchurPreconditioner::AlgebraicSchur: return SchurPreconditionerType::ALGEBRAIC_SHAT;
    }
    return SchurPreconditionerType::DIAG_L;
}

fe_fsi_linear_solver::SchurMomentumApproximationType
to_fsils_blockschur_momentum_approximation(FsilsBlockSchurMomentumApproximation approx)
{
    using fe_fsi_linear_solver::SchurMomentumApproximationType;
    switch (approx) {
        case FsilsBlockSchurMomentumApproximation::DiagK: return SchurMomentumApproximationType::DIAG_K;
        case FsilsBlockSchurMomentumApproximation::BlockDiagK: return SchurMomentumApproximationType::BLOCKDIAG_K;
        case FsilsBlockSchurMomentumApproximation::ILUK: return SchurMomentumApproximationType::ILU_K;
        case FsilsBlockSchurMomentumApproximation::ASM: return SchurMomentumApproximationType::ASM_K;
    }
    return SchurMomentumApproximationType::DIAG_K;
}

struct GmresLaunchConfig {
    int mItr{1};
    int sD{0};
};

[[nodiscard]] int gmres_expected_total_iterations(const GmresLaunchConfig& cfg)
{
    using i64 = long long;
    const i64 outer = static_cast<i64>(std::max(1, cfg.mItr));
    const i64 per_restart = static_cast<i64>(std::max(0, cfg.sD)) + 1LL;
    const i64 total = outer * per_restart;
    if (total > static_cast<i64>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(std::max<i64>(1, total));
}

[[nodiscard]] int gmres_total_iteration_budget(const SolverOptions& options,
                                               bool legacy_restart_budget)
{
    if (!legacy_restart_budget) {
        return std::max(1, options.max_iter);
    }

    // Match FSILS XML semantics used by the legacy solver:
    // - Max_iterations = restart count (mItr)
    // - Krylov_space_dimension = restart length (sD)
    const int restart_len = (options.krylov_dim > 0) ? options.krylov_dim : 250;
    using i64 = long long;
    const i64 outer = static_cast<i64>(std::max(1, options.max_iter));
    const i64 per_restart = static_cast<i64>(std::max(0, restart_len)) + 1LL;
    const i64 total = outer * per_restart;
    if (total > static_cast<i64>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(std::max<i64>(1, total));
}

[[nodiscard]] GmresLaunchConfig make_gmres_launch_config(const SolverOptions& options,
                                                         int requested_total_iterations,
                                                         int sD_override = 0)
{
    int sD = 0;
    if (sD_override > 0) {
        sD = sD_override;
    } else {
        const char* sd_env = std::getenv("SVMP_FSILS_GMRES_SD");
        if (sd_env) {
            try {
                sD = std::stoi(sd_env);
            } catch (...) {
                sD = 0;
            }
        }
    }
    if (sD <= 0) {
        sD = options.krylov_dim;
    }
    if (sD <= 0) {
        sD = std::max(0, std::min(250, requested_total_iterations) - 1);
    }

    const int total_iterations = std::max(1, requested_total_iterations);
    const int sD_max = std::max(0, total_iterations - 1);
    sD = std::clamp(sD, 0, sD_max);

    const int per_restart = sD + 1;
    const int mItr = std::max(1, (total_iterations + per_restart - 1) / std::max(1, per_restart));
    GmresLaunchConfig cfg;
    cfg.mItr = mItr;
    cfg.sD = sD;
    return cfg;
}

[[nodiscard]] int gmres_recovery_restart_dim(const GmresLaunchConfig& base_cfg,
                                             int requested_total_iterations)
{
    int recovery_sD = 0;
    if (const char* env = std::getenv("SVMP_FSILS_GMRES_RECOVERY_SD")) {
        try {
            recovery_sD = std::stoi(env);
        } catch (...) {
            recovery_sD = 0;
        }
    }
    if (recovery_sD <= 0) {
        recovery_sD = std::max(base_cfg.sD, 250);
    }
    const int sD_max = std::max(0, requested_total_iterations - 1);
    return std::clamp(recovery_sD, 0, sD_max);
}

[[nodiscard]] bool gmres_should_promote_restart(const SolverOptions& options,
                                                const FsilsResidualCheckResult& check,
                                                const GmresLaunchConfig& base_cfg,
                                                int actual_iterations)
{
    if (check.ok) {
        return false;
    }

    const Real rhs_norm = std::max<Real>(check.rhs_norm, Real(1e-30));
    const Real target = std::max<Real>(options.abs_tol, options.rel_tol * rhs_norm);
    if (!(target > Real(0.0)) || !std::isfinite(static_cast<double>(check.residual_norm))) {
        return true;
    }

    const Real miss_ratio = check.residual_norm / target;
    const int expected_iterations = gmres_expected_total_iterations(base_cfg);
    const int near_exhaustion = std::max(1, expected_iterations - std::max(1, base_cfg.sD + 1));
    const bool exhausted_budget = actual_iterations >= near_exhaustion;
    const bool severe_miss = miss_ratio > Real(25.0);
    return severe_miss && (exhausted_budget || miss_ratio > Real(100.0));
}

[[nodiscard]] bool gmres_retry_severe_stalls() noexcept
{
    if (const char* env = std::getenv("SVMP_FSILS_GMRES_RETRY_SEVERE_STALLS")) {
        const std::string value = env;
        if (value == "1" || value == "true" || value == "TRUE" ||
            value == "on" || value == "ON" || value == "yes" || value == "YES") {
            return true;
        }
    }
    return false;
}

[[nodiscard]] Real fsilsStrictValidationNearTargetFactor() noexcept
{
    Real factor = static_cast<Real>(4.0);
    if (const char* env = std::getenv("SVMP_FSILS_STRICT_VALIDATION_NEAR_TARGET_FACTOR")) {
        try {
            factor = static_cast<Real>(std::stod(env));
        } catch (...) {
            factor = static_cast<Real>(4.0);
        }
    }
    if (!std::isfinite(static_cast<double>(factor)) || factor < static_cast<Real>(1.0)) {
        factor = static_cast<Real>(4.0);
    }
    return factor;
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
    lhs.system_dof = dof;
    FE_THROW_IF(dof <= 0, FEException, "FsilsLinearSolver::solve: invalid FSILS dof");

    // Derive block structure from metadata. No physics-specific fallbacks —
    // all saddle-point operations require explicit block_layout with saddle-point annotation.
    const bool has_block_layout = options_.block_layout.has_value();
    const bool has_saddle_point = has_block_layout && options_.block_layout->hasSaddlePoint();

    // Saddle-point block indices (only meaningful when has_saddle_point is true).
    int mom_start = 0, mom_ncomp = 0;
    int con_start = 0, con_ncomp = 0;
    if (has_saddle_point) {
        const auto& layout = *options_.block_layout;
        const auto& mb = layout.blocks[static_cast<std::size_t>(*layout.momentum_block)];
        const auto& cb = layout.blocks[static_cast<std::size_t>(*layout.constraint_block)];
        mom_start = mb.start_component;
        mom_ncomp = mb.n_components;
        con_start = cb.start_component;
        con_ncomp = cb.n_components;
    }
    FE_THROW_IF(lhs.nNo <= 0, FEException, "FsilsLinearSolver::solve: invalid FSILS lhs.nNo");

    const GlobalIndex expected_local = static_cast<GlobalIndex>(lhs.nNo) * static_cast<GlobalIndex>(dof);
    FE_THROW_IF(static_cast<GlobalIndex>(x->data().size()) != expected_local ||
                    static_cast<GlobalIndex>(b->data().size()) != expected_local,
                FEException, "FsilsLinearSolver::solve: FSILS vectors must have local size lhs.nNo*dof");

    const bool requested_blockschur = (options_.method == SolverMethod::BlockSchur);
    const bool use_blockschur = requested_blockschur;

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

    if (requested_blockschur) {
        FE_THROW_IF(!has_saddle_point, NotImplementedException,
                    "FsilsLinearSolver::solve: BlockSchur requires block_layout with saddle-point annotation");
        const auto& layout = *options_.block_layout;
        const auto& mb = layout.blocks[static_cast<std::size_t>(*layout.momentum_block)];
        const auto& cb = layout.blocks[static_cast<std::size_t>(*layout.constraint_block)];
        FE_THROW_IF(mb.n_components < 1 || cb.n_components < 1,
                    NotImplementedException,
                    "FsilsLinearSolver::solve: BlockSchur requires valid saddle-point layout");
        FE_THROW_IF(mb.n_components + cb.n_components != dof,
                    NotImplementedException,
                    "FsilsLinearSolver::solve: BlockSchur saddle-point blocks must cover all DOFs "
                    "(momentum=" + std::to_string(mb.n_components) + " + constraint=" +
                    std::to_string(cb.n_components) + " != dof=" + std::to_string(dof) + ")");
    }

    // FSILS destructively modifies the matrix during preconditioning/solve.
    // Keep a solver-local copy for all methods so we can:
    // - validate the true post-solve residual against the original operator, and
    // - retry with a stricter Krylov configuration when the first solve is inexact.
    const GlobalIndex nnz = A->fsilsNnz();
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t value_count = static_cast<std::size_t>(nnz) * block_size;
    values_work_.resize(value_count);
    std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work_.data());

    // Public FE vectors stay in old local ordering, but the FSILS solve buffer is kept
    // in internal node ordering so overlap communication and fsils_solve() can operate
    // directly on the solver-native layout.
    auto& x_data = x->data();
    const auto& b_data = b->data();
    FE_THROW_IF(x_data.size() != b_data.size(), FEException, "FsilsLinearSolver::solve: RHS size mismatch");
    ri_internal_work_.resize(b_data.size());

    Array<double> Ri(dof, lhs.nNo, ri_internal_work_.data());
    FE_THROW_IF(nnz > static_cast<GlobalIndex>(std::numeric_limits<int>::max()), InvalidArgumentException,
                "FsilsLinearSolver::solve: nnz exceeds FSILS int index range");
    Array<double> Val(dof * dof, static_cast<int>(nnz), values_work_.data());

    // Optional scaling used for the BlockSchur solver path.
    //
    // The legacy solver scales resistance-type coupled BC tangent contributions by (gamma*dt),
    // where gamma is the generalized-α parameter. The OOP solver provides the effective stage
    // dt via LinearSolver::setEffectiveTimeStep().
    //
    // That transform is currently not robust for the distributed native BlockSchur path on the
    // coupled outlet application cases. Keep the algebra unscaled by default and leave the old
    // transform available only as an explicit diagnostic opt-in.
    double stage_scale = 1.0;
    if (use_blockschur && has_saddle_point) {
        if (std::getenv("SVMP_FSILS_ENABLE_BLOCKSCHUR_STAGE_SCALING") != nullptr &&
            std::getenv("SVMP_FSILS_DISABLE_BLOCKSCHUR_STAGE_SCALING") == nullptr &&
            std::isfinite(dt_eff_) && dt_eff_ > 0.0) {
            stage_scale = dt_eff_;
        }
    }

    auto applyStageScalingToMatrix = [&]() {
        if (stage_scale == 1.0) {
            return;
        }
        const Real s = static_cast<Real>(stage_scale);
        const Real inv_s = static_cast<Real>(1.0 / stage_scale);

        // Left-scale momentum rows.
        for (GlobalIndex bi = 0; bi < nnz; ++bi) {
            Real* blk = values_work_.data() + static_cast<std::size_t>(bi) * block_size;
            for (int r = mom_start; r < mom_start + mom_ncomp; ++r) {
                for (int c = 0; c < dof; ++c) {
                    blk[static_cast<std::size_t>(r * dof + c)] *= s;
                }
            }
            // Right-scale constraint columns to preserve G ≈ -D^T.
            for (int r = 0; r < dof; ++r) {
                for (int c = con_start; c < con_start + con_ncomp; ++c) {
                    blk[static_cast<std::size_t>(r * dof + c)] *= inv_s;
                }
            }
        }
    };

    auto restoreAndScaleMatrixValues = [&]() {
        std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work_.data());
        applyStageScalingToMatrix();
    };

    applyStageScalingToMatrix();

    const bool enforce_transpose_saddle_blocks =
        use_blockschur && has_saddle_point &&
        (std::getenv("SVMP_FSILS_ASSUME_TRANSPOSE_SADDLE") != nullptr) &&
        (std::getenv("SVMP_FSILS_DISABLE_TRANSPOSE_SADDLE") == nullptr);

    auto applySaddlePointEnforcement = [&]() {
        if (!enforce_transpose_saddle_blocks) {
            return;
        }
        const int nNo = lhs.nNo;
        const int nnz_int = lhs.nnz;
        if (nNo <= 0 || nnz_int <= 0) {
            return;
        }

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

        for (fe_fsi_linear_solver::fsils_int row = 0; row < nNo; ++row) {
            const auto start = lhs.rowPtr(0, row);
            const auto end = lhs.rowPtr(1, row);
            if (start < 0 || end < start) {
                continue;
            }
            for (auto idx = start; idx <= end; ++idx) {
                const auto col_idx = cols[idx];
                if (col_idx < 0 || col_idx >= nNo) {
                    continue;
                }

                const auto idx_t = find_entry(col_idx, row);
                if (idx_t < 0 || idx_t >= nnz_int) {
                    continue;
                }

                Real* blk = values_work_.data() + static_cast<std::size_t>(idx) * block_size;
                Real* blk_t = values_work_.data() + static_cast<std::size_t>(idx_t) * block_size;
                for (int vc = 0; vc < mom_ncomp; ++vc) {
                    for (int cc = 0; cc < con_ncomp; ++cc) {
                        const Real g_val = blk[static_cast<std::size_t>((mom_start + vc) * dof + (con_start + cc))];
                        blk_t[static_cast<std::size_t>((con_start + cc) * dof + (mom_start + vc))] = -g_val;
                    }
                }
            }
        }
    };

    if (enforce_transpose_saddle_blocks && oopTraceEnabled()) {
        traceLog("FsilsLinearSolver::solve: enforcing D=-G^T due to "
                 "SVMP_FSILS_ASSUME_TRANSPOSE_SADDLE.");
    }

    auto restorePreparedMatrixValues = [&](bool blockschur_preparation) {
        std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work_.data());
        if (blockschur_preparation) {
            applyStageScalingToMatrix();
            applySaddlePointEnforcement();
        }
    };

    std::vector<std::size_t> native_face_rank_one_indices;
    native_face_rank_one_indices.reserve(rank_one_updates_.size());
    for (std::size_t i = 0; i < rank_one_updates_.size(); ++i) {
        if (rank_one_updates_[i].prefer_native_face) {
            native_face_rank_one_indices.push_back(i);
        }
    }

    const bool has_native_rank_one_updates =
        !native_face_rank_one_indices.empty() || !reduced_field_updates_.empty();

    auto& ls = ls_;
    using BlockSchurStats = std::decay_t<decltype(ls.blockschur_stats)>;
    using SubSolverStats = std::decay_t<decltype(ls.GM.stats)>;
    BlockSchurStats preserved_blockschur_stats{};
    SubSolverStats preserved_blockschur_momentum_stats{};
    SubSolverStats preserved_blockschur_schur_stats{};
    bool preserved_blockschur_attempt = false;
    if (use_blockschur) {
        // FSILS BlockSchur uses RI.mItr as an outer basis dimension and allocates
        // O(nNo * mItr) workspace. Keep the legacy-safe cap for the fractional-step
        // path, then validate the true residual and retry with strict GMRES if needed.
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

        if (options_.fsils_blockschur_gm_max_iter) {
            ls.GM.mItr = *options_.fsils_blockschur_gm_max_iter;
        }
        if (options_.fsils_blockschur_cg_max_iter) {
            ls.CG.mItr = *options_.fsils_blockschur_cg_max_iter;
        }
        if (options_.fsils_blockschur_gm_rel_tol) {
            ls.GM.relTol = *options_.fsils_blockschur_gm_rel_tol;
        }
        if (options_.fsils_blockschur_cg_rel_tol) {
            ls.CG.relTol = *options_.fsils_blockschur_cg_rel_tol;
        }

        ls.RI.exact_convergence = true;
        ls.GM.exact_convergence = true;
        ls.CG.exact_convergence = true;
        ls.CG.schur_preconditioner =
            to_fsils_blockschur_preconditioner(options_.fsils_blockschur_schur_preconditioner);
        ls.CG.schur_momentum_approximation =
            to_fsils_blockschur_momentum_approximation(options_.fsils_blockschur_momentum_approximation);

        if (has_saddle_point) {
            ls.mom_start = mom_start;
            ls.mom_ncomp = mom_ncomp;
            ls.con_start = con_start;
            ls.con_ncomp = con_ncomp;
        }
    } else {
        const auto method = to_fsils_solver(options_.method);
        if (method == fe_fsi_linear_solver::LS_TYPE_GMRES) {
            const int gmres_total_iters = gmres_total_iteration_budget(options_, /*legacy_restart_budget=*/false);
            const auto gmres_cfg = make_gmres_launch_config(options_, gmres_total_iters);
            fe_fsi_linear_solver::fsils_ls_create(ls,
                                                  method,
                                                  options_.rel_tol,
                                                  options_.abs_tol,
                                                  gmres_cfg.mItr,
                                                  gmres_cfg.sD);
            // Native low-rank outlet corrections are unusually sensitive to
            // premature restarted-GMRES exits. Keep the heuristic early-stop
            // behavior for generic systems, but require exact convergence for
            // coupled rank-one solves so the first pass does not stop while the
            // true FE residual is still far from the requested tolerance.
            ls.RI.exact_convergence = has_native_rank_one_updates;
        } else {
            fe_fsi_linear_solver::fsils_ls_create(ls,
                                                  method,
                                                  options_.rel_tol,
                                                  options_.abs_tol,
                                                  options_.max_iter);
        }
    }
    ls.ri_internal_order = true;

    // Set up FSILS faces from:
    //  - Dirichlet constraints (legacy-equivalent FSILS preconditioner handling)
    //  - rank-1 updates (coupled BC Sherman-Morrison correction)
    const int original_nFaces = lhs.nFaces;
    const int num_dirichlet_faces = (!dirichlet_dofs_.empty() ? 1 : 0);
    const int num_rank_one_faces = static_cast<int>(native_face_rank_one_indices.size());
    const int num_added_faces = num_dirichlet_faces + num_rank_one_faces;

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
        fe_fsi_linear_solver::fsils_allreduce_sum(&local_has, &total_has, 1, MPI_INT, lhs.commu);

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

    // Face setup: restore from cache (fast path) or build from scratch.
    bool faces_from_cache = false;
    if (num_added_faces > 0 && dof > 0 && !faces_dirty_ &&
        cached_faces_.size() == static_cast<std::size_t>(num_added_faces)) {
        // Fast path: restore pre-built face data from cache.
        const int new_nFaces = original_nFaces + num_added_faces;
        lhs.face.resize(static_cast<std::size_t>(new_nFaces));
        lhs.nFaces = new_nFaces;

        int next_face = original_nFaces;
        if (num_dirichlet_faces > 0) dirichlet_face_index = next_face++;
        if (num_rank_one_faces > 0) rank_one_face_start = next_face;

        for (int fi = 0; fi < num_added_faces; ++fi) {
            const auto& cf = cached_faces_[static_cast<std::size_t>(fi)];
            auto& face = lhs.face[static_cast<std::size_t>(original_nFaces + fi)];
            face.nNo = cf.nNo;
            face.dof = cf.face_dof;
            face.bGrp = cf.bGrp;
            face.sharedFlag = cf.sharedFlag;
            face.foC = cf.foC;
            face.coupledFlag = cf.coupledFlag;
            face.incFlag = cf.incFlag;
            face.nS = 0.0;
            face.res = 0.0;
            if (cf.nNo > 0) {
                face.glob.resize(cf.nNo);
                std::copy(cf.glob_data.begin(), cf.glob_data.end(), face.glob.data());
                face.val.resize(cf.face_dof, cf.nNo);
                std::copy(cf.val_data.begin(), cf.val_data.end(), face.val.data());
                face.valM.resize(cf.face_dof, cf.nNo);
                std::copy(cf.valM_data.begin(), cf.valM_data.end(), face.valM.data());
            }
        }
        faces_from_cache = true;
    }

    lhs.native_face_rank_one_count = num_rank_one_faces;

    if (num_added_faces > 0 && dof > 0 && !faces_from_cache) {
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
        for (int u = 0; u < num_rank_one_faces; ++u) {
            const auto update_index = native_face_rank_one_indices[static_cast<std::size_t>(u)];
            const auto& upd = rank_one_updates_[update_index];
            const int faIn = rank_one_face_start + u;

            // Determine which per-node components participate in this rank-1 update.
            std::vector<int> face_comps;
            if (!upd.active_components.empty()) {
                face_comps = upd.active_components;
            } else if (has_saddle_point) {
                // Default: momentum block components only (skip constraint).
                for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                    face_comps.push_back(c);
                }
            } else {
                // No saddle-point info: all components participate.
                for (int c = 0; c < dof; ++c) {
                    face_comps.push_back(c);
                }
            }
            const int face_dof = static_cast<int>(face_comps.size());

            // Build a fast lookup: component index -> face-local index (-1 if not active).
            std::vector<int> comp_to_face_idx(static_cast<std::size_t>(dof), -1);
            for (int fi = 0; fi < face_dof; ++fi) {
                const int c = face_comps[static_cast<std::size_t>(fi)];
                if (c >= 0 && c < dof) {
                    comp_to_face_idx[static_cast<std::size_t>(c)] = fi;
                }
            }

            // Seed the overlap buffer from owned copies only, then synchronize
            // across shared nodes before materializing the face support. The
            // stored rank-1 update vector may come from a globally reduced
            // dQ/du, so seeding ghost copies here would double-count shared
            // outlet nodes under MPI.
            Array<double> face_values(face_dof, lhs.nNo);
            face_values = 0.0;
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

                // Only active components.
                if (comp < 0 || comp >= dof || comp_to_face_idx[static_cast<std::size_t>(comp)] < 0) {
                    continue;
                }

                const int old_local = shared->globalNodeToOld(node);
                if (old_local < 0 || old_local >= lhs.nNo) {
                    continue;
                }
                const int internal = lhs.map(old_local);
                if (internal < 0 || internal >= lhs.nNo) {
                    continue;
                }
                if (internal >= lhs.mynNo) {
                    continue;
                }
                face_values(comp_to_face_idx[static_cast<std::size_t>(comp)], internal) +=
                    static_cast<double>(val);
            }

            if (lhs.commu.nTasks > 1 && face_dof > 0) {
                fe_fsi_linear_solver::fsils_commuv(lhs, face_dof, face_values);
            }

            std::vector<int> face_nodes;
            face_nodes.reserve(static_cast<std::size_t>(lhs.nNo));
            for (int internal = 0; internal < lhs.nNo; ++internal) {
                bool has_support = false;
                for (int c = 0; c < face_dof; ++c) {
                    if (face_values(c, internal) != 0.0) {
                        has_support = true;
                        break;
                    }
                }
                if (has_support) {
                    face_nodes.push_back(internal);
                }
            }
            const int face_nNo = static_cast<int>(face_nodes.size());

            // Set up face data directly to avoid Vector/Array zero-size constructor issues.
            {
                auto& face = lhs.face[static_cast<std::size_t>(faIn)];
                face.nNo = face_nNo;
                face.dof = face_dof;
                face.bGrp = fe_fsi_linear_solver::BcType::BC_TYPE_Neu;

                if (face_nNo > 0) {
                    face.glob.resize(face_nNo);
                    face.val.resize(face_dof, face_nNo);
                    face.valM.resize(face_dof, face_nNo);
                    face.val = 0.0;
                    face.valM = 0.0;

                    for (int a = 0; a < face_nNo; ++a) {
                        const int internal = face_nodes[static_cast<std::size_t>(a)];
                        face.glob(a) = internal;
                        for (int c = 0; c < face_dof; ++c) {
                            face.val(c, a) = face_values(c, internal);
                        }
                    }
                }

                int local_has = (face_nNo > 0) ? 1 : 0;
                int total_has = local_has;
                if (lhs.commu.nTasks > 1) {
                    fe_fsi_linear_solver::fsils_allreduce_sum(&local_has, &total_has, 1, MPI_INT, lhs.commu);
                }
                face.sharedFlag = (total_has > 1);
                face.foC = true;
                face.coupledFlag = true;
                face.incFlag = true;
            }

            if (oopTraceEnabled()) {
                int owned_nodes = 0;
                int ghost_nodes = 0;
                const auto& face = lhs.face[static_cast<std::size_t>(faIn)];
                for (int a = 0; a < face.nNo; ++a) {
                    if (face.glob(a) < lhs.mynNo) {
                        ++owned_nodes;
                    } else {
                        ++ghost_nodes;
                    }
                }
                std::ostringstream oss;
                oss << "FsilsLinearSolver: rank-1 update " << update_index
                    << " -> FSILS face " << faIn
                    << " nNo=" << face_nNo
                    << " owned=" << owned_nodes
                    << " ghost=" << ghost_nodes
                    << " shared=" << (face.sharedFlag ? 1 : 0)
                    << " sigma=" << static_cast<double>(upd.sigma)
                    << " v_entries=" << upd.v.size();
                traceLog(oss.str());
            }
        }

        // Cache the built faces for reuse in subsequent Newton iterations.
        cached_faces_.clear();
        cached_faces_.resize(static_cast<std::size_t>(num_added_faces));
        for (int fi = 0; fi < num_added_faces; ++fi) {
            const auto& face = lhs.face[static_cast<std::size_t>(original_nFaces + fi)];
            auto& cf = cached_faces_[static_cast<std::size_t>(fi)];
            cf.nNo = face.nNo;
            cf.face_dof = face.dof;
            cf.bGrp = face.bGrp;
            cf.sharedFlag = face.sharedFlag;
            cf.foC = face.foC;
            cf.coupledFlag = face.coupledFlag;
            cf.incFlag = face.incFlag;
            if (face.nNo > 0) {
                const auto sz = static_cast<std::size_t>(face.dof) * static_cast<std::size_t>(face.nNo);
                cf.glob_data.assign(face.glob.data(), face.glob.data() + face.nNo);
                cf.val_data.assign(face.val.data(), face.val.data() + sz);
                cf.valM_data.assign(face.valM.data(), face.valM.data() + sz);
            }
        }
        faces_dirty_ = false;
    }

    lhs.reduced_updates.clear();
    lhs.grouped_bordered_field_couplings.clear();
    {
        const auto shared = A->shared();
        FE_CHECK_NOT_NULL(shared.get(), "FsilsLinearSolver: FsilsShared for reduced updates");

        auto default_active_components = [&]() {
            std::vector<int> comps;
            if (has_saddle_point) {
                comps.reserve(static_cast<std::size_t>(mom_ncomp));
                for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                    comps.push_back(c);
                }
            } else {
                comps.reserve(static_cast<std::size_t>(dof));
                for (int c = 0; c < dof; ++c) {
                    comps.push_back(c);
                }
            }
            return comps;
        };

        auto make_internal_entries =
            [&](std::span<const std::pair<GlobalIndex, Real>> entries)
                -> std::pair<std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>,
                             std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>> {
            Array<double> values(dof, lhs.nNo);
            values = 0.0;

            for (const auto& [dof_idx, val] : entries) {
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
                const int internal = lhs.map(old_local);
                if (internal < 0 || internal >= lhs.nNo || internal >= lhs.mynNo) {
                    continue;
                }
                values(comp, internal) += static_cast<double>(val);
            }

            if (lhs.commu.nTasks > 1) {
                fe_fsi_linear_solver::fsils_commuv(lhs, dof, values);
            }

            std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry> full;
            std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry> owned;
            full.reserve(static_cast<std::size_t>(dof) * static_cast<std::size_t>(lhs.nNo));
            owned.reserve(static_cast<std::size_t>(dof) * static_cast<std::size_t>(lhs.mynNo));
            for (int internal = 0; internal < lhs.nNo; ++internal) {
                for (int comp = 0; comp < dof; ++comp) {
                    const double value = values(comp, internal);
                    if (std::abs(value) <= 1e-30) {
                        continue;
                    }
                    fe_fsi_linear_solver::FSILS_reducedSparseEntry entry;
                    entry.node = static_cast<fe_fsi_linear_solver::fsils_int>(internal);
                    entry.full_component = comp;
                    entry.value = value;
                    full.push_back(entry);
                    if (internal < lhs.mynNo) {
                        owned.push_back(entry);
                    }
                }
            }
            return {std::move(full), std::move(owned)};
        };

        auto make_native_reduced_update = [&](Real sigma,
                                              std::span<const std::pair<GlobalIndex, Real>> left,
                                              std::span<const std::pair<GlobalIndex, Real>> right,
                                              std::span<const int> active_components,
                                              int grouped_coupling_id,
                                              Real left_scale,
                                              bool scale_sigma) {
            fe_fsi_linear_solver::FSILS_reducedFieldUpdateType native_update;
            if (!(std::abs(sigma) > Real(1e-30))) {
                return native_update;
            }

            auto [left_full, left_owned] = make_internal_entries(left);
            auto [right_full, right_owned] = make_internal_entries(right);
            int local_left_has = left_owned.empty() ? 0 : 1;
            int local_right_has = right_owned.empty() ? 0 : 1;
            int global_left_has = local_left_has;
            int global_right_has = local_right_has;
            if (lhs.commu.nTasks > 1) {
                fe_fsi_linear_solver::fsils_allreduce_sum(
                    &local_left_has, &global_left_has, 1, MPI_INT, lhs.commu);
                fe_fsi_linear_solver::fsils_allreduce_sum(
                    &local_right_has, &global_right_has, 1, MPI_INT, lhs.commu);
            }
            if (global_left_has == 0 || global_right_has == 0) {
                return native_update;
            }

            if (left_scale != Real(1.0)) {
                for (auto& entry : left_full) {
                    entry.value *= static_cast<double>(left_scale);
                }
                for (auto& entry : left_owned) {
                    entry.value *= static_cast<double>(left_scale);
                }
            }

            native_update.active = true;
            native_update.sigma = static_cast<double>(
                (scale_sigma && use_blockschur) ? sigma * stage_scale : sigma);
            native_update.grouped_coupling_id = grouped_coupling_id;
            native_update.left = std::move(left_full);
            native_update.right = std::move(right_full);
            native_update.left_owned = std::move(left_owned);
            native_update.right_owned = std::move(right_owned);
            native_update.left_scaled = native_update.left;
            native_update.right_scaled = native_update.right;
            native_update.left_scaled_owned = native_update.left_owned;
            native_update.right_scaled_owned = native_update.right_owned;
            if (!active_components.empty()) {
                native_update.active_components.assign(active_components.begin(), active_components.end());
            } else {
                native_update.active_components = default_active_components();
            }
            return native_update;
        };

        auto build_face_from_reduced_entries =
            [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
                const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& entries,
                fe_fsi_linear_solver::FSILS_faceType& face) {
            std::vector<int> face_comps = !update.active_components.empty()
                                              ? update.active_components
                                              : default_active_components();
            const int face_dof = static_cast<int>(face_comps.size());
            if (face_dof <= 0) {
                return;
            }

            std::vector<int> comp_to_face_idx(static_cast<std::size_t>(dof), -1);
            for (int fi = 0; fi < face_dof; ++fi) {
                const int comp = face_comps[static_cast<std::size_t>(fi)];
                if (comp >= 0 && comp < dof) {
                    comp_to_face_idx[static_cast<std::size_t>(comp)] = fi;
                }
            }

            Array<double> face_values(face_dof, lhs.nNo);
            face_values = 0.0;
            for (const auto& entry : entries) {
                if (entry.node < 0 || entry.node >= lhs.nNo || std::abs(entry.value) <= 1e-30) {
                    continue;
                }
                if (entry.full_component < 0 || entry.full_component >= dof) {
                    continue;
                }
                const int face_comp = comp_to_face_idx[static_cast<std::size_t>(entry.full_component)];
                if (face_comp < 0) {
                    continue;
                }
                face_values(face_comp, entry.node) += entry.value;
            }

            std::vector<int> face_nodes;
            face_nodes.reserve(static_cast<std::size_t>(lhs.nNo));
            for (int internal = 0; internal < lhs.nNo; ++internal) {
                bool has_support = false;
                for (int c = 0; c < face_dof; ++c) {
                    if (std::abs(face_values(c, internal)) > 0.0) {
                        has_support = true;
                        break;
                    }
                }
                if (has_support) {
                    face_nodes.push_back(internal);
                }
            }

            face.nNo = static_cast<int>(face_nodes.size());
            face.dof = face_dof;
            face.bGrp = fe_fsi_linear_solver::BcType::BC_TYPE_Neu;
            face.foC = true;
            face.coupledFlag = true;
            face.incFlag = true;
            face.sharedFlag = false;
            face.nS = 0.0;
            face.res = 0.0;
            if (face.nNo > 0) {
                face.glob.resize(face.nNo);
                face.val.resize(face_dof, face.nNo);
                face.valM.resize(face_dof, face.nNo);
                face.val = 0.0;
                face.valM = 0.0;
                for (int a = 0; a < face.nNo; ++a) {
                    const int internal = face_nodes[static_cast<std::size_t>(a)];
                    face.glob(a) = internal;
                    for (int c = 0; c < face_dof; ++c) {
                        face.val(c, a) = face_values(c, internal);
                        face.valM(c, a) = face_values(c, internal);
                    }
                }
            }

            sort_face_by_glob(face, face_dof);
            sync_face_val_if_shared(face, face_dof);
        };

        auto append_reduced_update = [&](Real sigma,
                                        std::span<const std::pair<GlobalIndex, Real>> left,
                                        std::span<const std::pair<GlobalIndex, Real>> right,
                                        std::span<const int> active_components,
                                        int grouped_coupling_id) {
            auto native_update =
                make_native_reduced_update(sigma, left, right, active_components,
                                           grouped_coupling_id, Real(1.0),
                                           /*scale_sigma=*/true);
            if (native_update.active) {
                build_face_from_reduced_entries(native_update, native_update.left, native_update.left_face);
                build_face_from_reduced_entries(native_update, native_update.right, native_update.right_face);
                native_update.has_face_cache =
                    native_update.left_face.nNo > 0 && native_update.right_face.nNo > 0;
                lhs.reduced_updates.push_back(std::move(native_update));
            }
        };

        for (const auto& upd : rank_one_updates_) {
            if (upd.prefer_native_face) {
                continue;
            }
            append_reduced_update(upd.sigma,
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.v.data(), upd.v.size()),
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.v.data(), upd.v.size()),
                                  std::span<const int>(upd.active_components.data(),
                                                       upd.active_components.size()),
                                  /*grouped_coupling_id=*/-1);
        }

        for (const auto& upd : reduced_field_updates_) {
            append_reduced_update(upd.sigma,
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.left.data(), upd.left.size()),
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.right.data(), upd.right.size()),
                                  std::span<const int>(upd.active_components.data(),
                                                       upd.active_components.size()),
                                  upd.grouped_coupling_id);
        }

        const Real grouped_left_scale = use_blockschur ? static_cast<Real>(stage_scale) : Real(1.0);
        for (const auto& group : grouped_bordered_field_couplings_) {
            fe_fsi_linear_solver::FSILS_groupedBorderedFieldCouplingType native_group;
            native_group.active = true;
            native_group.grouped_coupling_id = group.grouped_coupling_id;
            native_group.aux_matrix.assign(group.aux_matrix.begin(), group.aux_matrix.end());
            native_group.modes.reserve(group.modes.size());
            native_group.left_faces.reserve(group.modes.size());
            native_group.right_faces.reserve(group.modes.size());
            for (const auto& mode : group.modes) {
                auto native_mode =
                    make_native_reduced_update(Real(1.0),
                                               std::span<const std::pair<GlobalIndex, Real>>(
                                                   mode.left.data(), mode.left.size()),
                                               std::span<const std::pair<GlobalIndex, Real>>(
                                                   mode.right.data(), mode.right.size()),
                                               std::span<const int>(mode.active_components.data(),
                                                                    mode.active_components.size()),
                                               group.grouped_coupling_id,
                                               grouped_left_scale,
                                               /*scale_sigma=*/false);
                if (native_mode.active) {
                    fe_fsi_linear_solver::FSILS_faceType left_face;
                    fe_fsi_linear_solver::FSILS_faceType right_face;
                    build_face_from_reduced_entries(native_mode, native_mode.left, left_face);
                    build_face_from_reduced_entries(native_mode, native_mode.right, right_face);
                    native_group.modes.push_back(std::move(native_mode));
                    native_group.left_faces.push_back(std::move(left_face));
                    native_group.right_faces.push_back(std::move(right_face));
                }
            }
            if (!native_group.aux_matrix.empty() && !native_group.modes.empty()) {
                lhs.grouped_bordered_field_couplings.push_back(std::move(native_group));
            }
        }
    }

    // Build incL and res vectors for face activation.
    // When no faces exist, pass empty vectors (original behavior).
    // Note: must use default constructors, not Vector(0), because Vector(0) leaves
    // data_ uninitialized (legacy Fortran compat), causing crashes in resize().
    Vector<int> incL;
    Vector<double> res_original;
    Vector<double> res_blockschur;
    if (lhs.nFaces > 0) {
        const int total_faces = lhs.nFaces;
        incL.resize(total_faces);
        res_original.resize(total_faces);
        res_blockschur.resize(total_faces);
        for (int f = 0; f < total_faces; ++f) {
            incL(f) = 1;
            res_original(f) = 0.0;
            res_blockschur(f) = 0.0;
        }
        if (num_rank_one_faces > 0 && rank_one_face_start >= 0) {
            // Set resistance values for rank-1 faces.
            for (int u = 0; u < num_rank_one_faces; ++u) {
                const int faIn = rank_one_face_start + u;
                const auto update_index = native_face_rank_one_indices[static_cast<std::size_t>(u)];
                const double sigma = static_cast<double>(rank_one_updates_[update_index].sigma);
                res_original(faIn) = sigma;
                res_blockschur(faIn) = sigma * stage_scale;
            }
        }
    }

    const auto prec = to_fsils_prec(options_);

    double rhs_prepare_time_seconds = 0.0;
    double validation_time_seconds = 0.0;
    const auto solve_buffer = std::span<Real>(ri_internal_work_.data(), ri_internal_work_.size());

    auto loadSolveBufferFromVector = [&](const FsilsVector& src, bool blockschur_preparation) {
        const double tp0 = fe_fsi_linear_solver::fsils_cpu_t();
        copyVectorOldToInternal(src, solve_buffer);
        fe_fsi_linear_solver::fsils_commuv(lhs, dof, Ri);

        if (blockschur_preparation && stage_scale != 1.0) {
            const Real s = static_cast<Real>(stage_scale);
            for (int a = 0; a < lhs.nNo; ++a) {
                // Scale momentum rows of the RHS.
                for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                    Ri(c, a) *= s;
                }
            }
        }
        rhs_prepare_time_seconds += fe_fsi_linear_solver::fsils_cpu_t() - tp0;
    };

    auto storeSolveBufferToSolution = [&]() {
        copyVectorInternalToOld(std::span<const Real>(solve_buffer.data(), solve_buffer.size()), *x);
    };

    SolverReport report;
    const int base_gmres_total_iterations =
        gmres_total_iteration_budget(options_, /*legacy_restart_budget=*/false);
    const auto base_gmres_cfg = make_gmres_launch_config(options_, base_gmres_total_iterations);
    bool solution_stage_scaling_undone = false;
    bool current_preparation_uses_blockschur = use_blockschur;

    auto rebuildPreparedSystem = [&](bool blockschur_preparation) {
        restorePreparedMatrixValues(blockschur_preparation);
        loadSolveBufferFromVector(*b, blockschur_preparation);
        solution_stage_scaling_undone = false;
        current_preparation_uses_blockschur = blockschur_preparation;
        if (oopTraceEnabled()) {
            traceLog(std::string("FsilsLinearSolver::solve: prepared system mode='") +
                     (blockschur_preparation ? "blockschur" : "original") + "'");
        }
    };

    const auto shared_layout = A->shared();
    FE_CHECK_NOT_NULL(shared_layout.get(), "FsilsLinearSolver::solve: shared layout");

    auto computeTrueResidualVector = [&](FsilsVector& residual_true, Real& rhs_norm_out) {
        FsilsVector rhs_true(shared_layout);
        rhs_true.copyFrom(*b);
        rhs_true.accumulateOverlap();
        rhs_norm_out = rhs_true.norm();

        FsilsVector x_true(shared_layout);
        x_true.copyFrom(*x);
        switch (fsilsResidualValidationSyncMode()) {
            case FsilsResidualValidationSyncMode::UpdateGhosts:
                x_true.updateGhosts();
                break;
            case FsilsResidualValidationSyncMode::AccumulateOverlap:
                x_true.accumulateOverlap();
                break;
            case FsilsResidualValidationSyncMode::AccumulateThenUpdateGhosts:
                x_true.accumulateOverlap();
                x_true.updateGhosts();
                break;
        }

        FsilsVector ax_true(shared_layout);
        A->mult(x_true, ax_true);
        addRankOneUpdatesToProduct(rank_one_updates_, x_true, ax_true, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_, x_true, ax_true, lhs.commu);

        residual_true.copyFrom(rhs_true);
        auto r_span = residual_true.localSpan();
        const auto ax_span = ax_true.localSpan();
        FE_THROW_IF(r_span.size() != ax_span.size(), FEException,
                    "FsilsLinearSolver::solve: residual validation size mismatch");
        for (std::size_t i = 0; i < r_span.size(); ++i) {
            r_span[i] -= ax_span[i];
        }
    };

    auto computeOriginalRhsNorm = [&]() -> Real {
        FsilsVector rhs_true(shared_layout);
        rhs_true.copyFrom(*b);
        rhs_true.accumulateOverlap();
        return rhs_true.norm();
    };

    auto computeConstraintMeanStats = [&]() -> FsilsConstraintMeanStats {
        FsilsConstraintMeanStats stats;
        if (!has_saddle_point || con_ncomp != 1 || con_start < 0 || con_start >= dof) {
            return stats;
        }

        long double local_sum = 0.0L;
        long double local_sq = 0.0L;
        unsigned long long local_count = 0ull;
        for (int a = 0; a < lhs.mynNo; ++a) {
            const Real value = Ri(con_start, a);
            local_sum += static_cast<long double>(value);
            local_sq += static_cast<long double>(value) * static_cast<long double>(value);
            ++local_count;
        }

        long double global_sum = local_sum;
        long double global_sq = local_sq;
        unsigned long long global_count = local_count;
        if (lhs.commu.nTasks > 1) {
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sum, &global_sum, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sq, &global_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, lhs.commu);
        }

        if (global_count == 0ull) {
            return stats;
        }

        const long double inv_count = 1.0L / static_cast<long double>(global_count);
        const long double mean = global_sum * inv_count;
        const long double mean_sq = mean * mean;
        const long double rms_sq = std::max<long double>(0.0L, global_sq * inv_count);
        const long double fluct_sq = std::max<long double>(0.0L, rms_sq - mean_sq);

        stats.valid = std::isfinite(static_cast<double>(mean)) &&
                      std::isfinite(static_cast<double>(rms_sq)) &&
                      std::isfinite(static_cast<double>(fluct_sq));
        stats.count = static_cast<std::uint64_t>(global_count);
        stats.mean = static_cast<Real>(mean);
        stats.rms = static_cast<Real>(std::sqrt(rms_sq));
        stats.fluctuation_rms = static_cast<Real>(std::sqrt(fluct_sq));
        return stats;
    };

    auto subtractConstraintMean = [&](Real mean_shift) {
        if (mean_shift == Real(0.0)) {
            return;
        }
        for (int a = 0; a < lhs.nNo; ++a) {
            Ri(con_start, a) -= mean_shift;
        }
    };

    auto computeReturnedSolutionConstraintMeanStats = [&]() -> FsilsConstraintMeanStats {
        FsilsConstraintMeanStats stats;
        if (!has_saddle_point || con_ncomp != 1 || con_start < 0 || con_start >= dof) {
            return stats;
        }

        long double local_sum = 0.0L;
        long double local_sq = 0.0L;
        unsigned long long local_count = 0ull;
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                static_cast<std::size_t>(con_start);
            const Real value = x_data[idx];
            local_sum += static_cast<long double>(value);
            local_sq += static_cast<long double>(value) * static_cast<long double>(value);
            ++local_count;
        }

        long double global_sum = local_sum;
        long double global_sq = local_sq;
        unsigned long long global_count = local_count;
        if (lhs.commu.nTasks > 1) {
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sum, &global_sum, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sq, &global_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, lhs.commu);
        }

        if (global_count == 0ull) {
            return stats;
        }

        const long double inv_count = 1.0L / static_cast<long double>(global_count);
        const long double mean = global_sum * inv_count;
        const long double mean_sq = mean * mean;
        const long double rms_sq = std::max<long double>(0.0L, global_sq * inv_count);
        const long double fluct_sq = std::max<long double>(0.0L, rms_sq - mean_sq);

        stats.valid = std::isfinite(static_cast<double>(mean)) &&
                      std::isfinite(static_cast<double>(rms_sq)) &&
                      std::isfinite(static_cast<double>(fluct_sq));
        stats.count = static_cast<std::uint64_t>(global_count);
        stats.mean = static_cast<Real>(mean);
        stats.rms = static_cast<Real>(std::sqrt(rms_sq));
        stats.fluctuation_rms = static_cast<Real>(std::sqrt(fluct_sq));
        return stats;
    };

    auto subtractConstraintMeanFromReturnedSolution = [&](Real mean_shift) {
        if (mean_shift == Real(0.0)) {
            return;
        }
        for (int old = 0; old < lhs.nNo; ++old) {
            const std::size_t idx =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                static_cast<std::size_t>(con_start);
            x_data[idx] -= mean_shift;
        }
    };

    auto centerReturnedSolutionConstraintMean = [&](std::string_view phase,
                                                    Real dominance_threshold,
                                                    bool force) -> bool {
        if (!(has_native_rank_one_updates && has_saddle_point && con_ncomp == 1)) {
            return false;
        }

        const auto before = computeReturnedSolutionConstraintMeanStats();
        if (!before.valid || before.count == 0u) {
            return false;
        }

        const Real fluctuation_floor =
            std::max<Real>(before.rms * static_cast<Real>(1e-12), static_cast<Real>(1e-14));
        const Real fluctuation = std::max(before.fluctuation_rms, fluctuation_floor);
        const Real dominance = std::abs(before.mean) / fluctuation;
        const bool should_center =
            force || (dominance >= dominance_threshold && std::abs(before.mean) > static_cast<Real>(1e-14));
        if (!should_center) {
            return false;
        }

        subtractConstraintMeanFromReturnedSolution(before.mean);
        if (oopTraceEnabled()) {
            const auto after = computeReturnedSolutionConstraintMeanStats();
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: centered returned constraint mean"
                << " phase='" << phase << "'"
                << " mean_before=" << before.mean
                << " rms_before=" << before.rms
                << " fluct_before=" << before.fluctuation_rms
                << " dominance=" << dominance
                << " mean_after=" << after.mean
                << " rms_after=" << after.rms;
            traceLog(oss.str());
        }
        return true;
    };

    auto logInternalBlockSolutionStats = [&](std::string_view phase) {
        if (!oopTraceEnabled() || !has_saddle_point) {
            return;
        }

        long double local_mom_sq = 0.0L;
        long double local_con_sq = 0.0L;
        for (int a = 0; a < lhs.mynNo; ++a) {
            for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                const long double v = static_cast<long double>(Ri(c, a));
                local_mom_sq += v * v;
            }
            for (int c = con_start; c < con_start + con_ncomp; ++c) {
                const long double v = static_cast<long double>(Ri(c, a));
                local_con_sq += v * v;
            }
        }

        long double global_mom_sq = local_mom_sq;
        long double global_con_sq = local_con_sq;
        if (lhs.commu.nTasks > 1) {
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_mom_sq, &global_mom_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_con_sq, &global_con_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
        }

        const auto correction_con_stats = computeConstraintMeanStats();
        const auto returned_con_stats = computeReturnedSolutionConstraintMeanStats();
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: internal block stats"
            << " phase='" << phase << "'"
            << " |mom|=" << std::sqrt(std::max<long double>(0.0L, global_mom_sq))
            << " |con|=" << std::sqrt(std::max<long double>(0.0L, global_con_sq));
        if (correction_con_stats.valid) {
            oss << " corr_con_mean=" << correction_con_stats.mean
                << " corr_con_rms=" << correction_con_stats.rms
                << " corr_con_fluct=" << correction_con_stats.fluctuation_rms;
        }
        if (returned_con_stats.valid) {
            oss << " returned_con_mean=" << returned_con_stats.mean
                << " returned_con_rms=" << returned_con_stats.rms
                << " returned_con_fluct=" << returned_con_stats.fluctuation_rms;
        }
        traceLog(oss.str());
    };

    auto compareFaceOperatorAgainstFe = [&]() {
        if (!fsilsCompareFaceOperatorEnabled() ||
            (rank_one_updates_.empty() && reduced_field_updates_.empty())) {
            return;
        }

        FsilsVector probe_old(shared_layout);
        {
            auto probe_span = probe_old.localSpan();
            for (int old = 0; old < lhs.nNo; ++old) {
                const int global_node = shared_layout->oldToGlobalNode(old);
                const std::size_t base =
                    static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                for (int c = 0; c < dof; ++c) {
                    probe_span[base + static_cast<std::size_t>(c)] =
                        static_cast<Real>(0.001 * (static_cast<double>(global_node * dof + c) + 1.0));
                }
            }
        }
        probe_old.updateGhosts();

        FsilsVector fe_y(shared_layout);
        A->mult(probe_old, fe_y);
        addRankOneUpdatesToProduct(rank_one_updates_, probe_old, fe_y, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_, probe_old, fe_y, lhs.commu);

        std::vector<Real> probe_internal_data(solve_buffer.size(), Real(0.0));
        std::vector<Real> fsils_internal_data(solve_buffer.size(), Real(0.0));
        copyVectorOldToInternal(probe_old, std::span<Real>(probe_internal_data.data(), probe_internal_data.size()));
        Array<double> probe_internal(dof, lhs.nNo, probe_internal_data.data());
        Array<double> fsils_internal(dof, lhs.nNo, fsils_internal_data.data());
        fe_fsi_linear_solver::fsils_commuv(lhs, dof, probe_internal);

        spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, probe_internal, fsils_internal);
        add_bc_mul::add_bc_mul(lhs, fe_fsi_linear_solver::BcopType::BCOP_TYPE_ADD,
                               dof, probe_internal, fsils_internal);

        FsilsVector fsils_y(shared_layout);
        copyVectorInternalToOld(std::span<const Real>(fsils_internal_data.data(), fsils_internal_data.size()), fsils_y);

        FsilsVector diff(shared_layout);
        diff.copyFrom(fe_y);
        auto diff_span = diff.localSpan();
        const auto fsils_span = fsils_y.localSpan();
        FE_THROW_IF(diff_span.size() != fsils_span.size(), FEException,
                    "FsilsLinearSolver::solve: operator compare size mismatch");
        for (std::size_t i = 0; i < diff_span.size(); ++i) {
            diff_span[i] -= fsils_span[i];
        }

        const Real fe_norm = fe_y.norm();
        const Real fsils_norm = fsils_y.norm();
        const Real diff_norm = diff.norm();
        const Real rel = diff_norm / std::max<Real>(fe_norm, Real(1e-30));
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: face operator compare"
                << " |FE|=" << fe_norm
                << " |FSILS|=" << fsils_norm
                << " |diff|=" << diff_norm
                << " rel=" << rel
                << " reduced_updates=" << (rank_one_updates_.size() + reduced_field_updates_.size());
            traceLog(oss.str());
        }
    };

    auto logReturnedOperatorBreakdown = [&](std::string_view phase, const FsilsVector& solution) {
        if (!oopTraceEnabled()) {
            return;
        }

        FsilsVector rhs_eval(shared_layout);
        rhs_eval.copyFrom(*b);
        rhs_eval.accumulateOverlap();

        FsilsVector x_eval(shared_layout);
        x_eval.copyFrom(solution);
        x_eval.updateGhosts();

        FsilsVector matrix_eval(shared_layout);
        A->mult(x_eval, matrix_eval);

        FsilsVector matrix_residual(shared_layout);
        matrix_residual.copyFrom(matrix_eval);
        {
            auto matrix_residual_span = matrix_residual.localSpan();
            const auto rhs_span = rhs_eval.localSpan();
            FE_THROW_IF(matrix_residual_span.size() != rhs_span.size(),
                        FEException,
                        "FsilsLinearSolver::solve: returned matrix residual size mismatch");
            for (std::size_t i = 0; i < matrix_residual_span.size(); ++i) {
                matrix_residual_span[i] -= rhs_span[i];
            }
        }

        FsilsVector rank_one_eval(shared_layout);
        rank_one_eval.zero();
        addRankOneUpdatesToProduct(rank_one_updates_, x_eval, rank_one_eval, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_, x_eval, rank_one_eval, lhs.commu);

        FsilsVector full_residual(shared_layout);
        full_residual.copyFrom(matrix_residual);
        {
            auto full_residual_span = full_residual.localSpan();
            const auto rank_one_span = rank_one_eval.localSpan();
            FE_THROW_IF(full_residual_span.size() != rank_one_span.size(),
                        FEException,
                        "FsilsLinearSolver::solve: returned full residual size mismatch");
            for (std::size_t i = 0; i < full_residual_span.size(); ++i) {
                full_residual_span[i] += rank_one_span[i];
            }
        }

        std::vector<Real> dots(rank_one_updates_.size(), Real(0.0));
        if (!rank_one_updates_.empty()) {
            auto x_view = x_eval.createAssemblyView();
            FE_CHECK_NOT_NULL(x_view.get(), "FsilsLinearSolver::solve: returned x view");
            for (std::size_t u = 0; u < rank_one_updates_.size(); ++u) {
                Real dot = Real(0.0);
                for (const auto& [dof, val] : rank_one_updates_[u].v) {
                    dot += val * x_view->getVectorEntry(dof);
                }
#if FE_HAS_MPI
                int mpi_initialized = 0;
                MPI_Initialized(&mpi_initialized);
                if (mpi_initialized && lhs.commu.nTasks > 1) {
                    Real global_dot = Real(0.0);
                    fe_fsi_linear_solver::fsils_allreduce_sum(
                        &dot, &global_dot, 1, MPI_DOUBLE, lhs.commu);
                    dot = global_dot;
                }
#endif
                dots[u] = dot;
            }
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: returned operator breakdown"
            << " phase='" << phase << "'"
            << " |x|=" << x_eval.norm()
            << " |Jx|=" << matrix_eval.norm()
            << " |Jx-r|=" << matrix_residual.norm()
            << " |rank1*x|=" << rank_one_eval.norm()
            << " |(J+R)x-r|=" << full_residual.norm();
        for (std::size_t u = 0; u < dots.size(); ++u) {
            oss << " dot[" << u << "]=" << dots[u];
        }
        traceLog(oss.str());
    };

    auto validateOriginalResidual = [&](std::string_view phase) -> FsilsResidualCheckResult {
        const double tp0 = fe_fsi_linear_solver::fsils_cpu_t();
        FsilsResidualCheckResult result;
        FsilsVector residual_true(shared_layout);
        computeTrueResidualVector(residual_true, result.rhs_norm);

        result.residual_norm = residual_true.norm();
        const Real denom = std::max<Real>(result.rhs_norm, 1e-30);
        result.relative_residual = result.residual_norm / denom;
        const Real target = std::max<Real>(options_.abs_tol, options_.rel_tol * denom);
        const bool finite = std::isfinite(static_cast<double>(result.residual_norm)) &&
                            std::isfinite(static_cast<double>(result.relative_residual));
        result.ok = finite && result.residual_norm <= target;
        if (!result.ok) {
            std::ostringstream oss;
            oss << phase << ": true residual check failed (|Ax-b|=" << result.residual_norm
                << ", rel=" << result.relative_residual
                << ", target=" << target << ")";
            result.detail = oss.str();
        }
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: true residual"
                << " phase='" << phase << "'"
                << " ok=" << (result.ok ? 1 : 0)
                << " |Ax-b|=" << result.residual_norm
                << " rel=" << result.relative_residual
                << " rhs=" << result.rhs_norm;
            if (!result.detail.empty()) {
                oss << " detail='" << result.detail << "'";
            }
            traceLog(oss.str());
        }
        validation_time_seconds += fe_fsi_linear_solver::fsils_cpu_t() - tp0;
        return result;
    };

    auto maybeRecenterConstraintMeanAndValidate =
        [&](std::string_view phase,
            const FsilsResidualCheckResult& baseline_check) -> FsilsResidualCheckResult {
        if (!(has_native_rank_one_updates && has_saddle_point && con_ncomp == 1)) {
            return baseline_check;
        }

        const auto before = computeReturnedSolutionConstraintMeanStats();
        if (!before.valid || before.count == 0u) {
            return baseline_check;
        }
        if (!(std::abs(before.mean) > static_cast<Real>(1e-14))) {
            return baseline_check;
        }

        const Real fluctuation_floor =
            std::max<Real>(before.rms * static_cast<Real>(1e-12), static_cast<Real>(1e-14));
        const Real fluctuation = std::max(before.fluctuation_rms, fluctuation_floor);
        const Real dominance = std::abs(before.mean) / fluctuation;

        std::vector<Real> backup = x_data;
        subtractConstraintMeanFromReturnedSolution(before.mean);
        auto recentered_check = validateOriginalResidual(std::string(phase) + "_recentered");
        const auto after = computeReturnedSolutionConstraintMeanStats();

        const bool baseline_finite = std::isfinite(static_cast<double>(baseline_check.residual_norm));
        const bool recentered_finite = std::isfinite(static_cast<double>(recentered_check.residual_norm));
        const Real target = std::max<Real>(
            options_.abs_tol, options_.rel_tol * std::max<Real>(recentered_check.rhs_norm, Real(1e-30)));
        const bool residual_not_worse =
            recentered_finite &&
            (!baseline_finite ||
             recentered_check.residual_norm <=
                 std::max<Real>(baseline_check.residual_norm * static_cast<Real>(1.05),
                                baseline_check.residual_norm + target));
        const bool mean_removed =
            after.valid &&
            std::abs(after.mean) <= std::max<Real>(std::abs(before.mean) * static_cast<Real>(1e-6),
                                                   static_cast<Real>(1e-10));
        const bool accept = mean_removed && (recentered_check.ok || residual_not_worse);

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: constraint recenter"
                << " phase='" << phase << "'"
                << " mean_before=" << before.mean
                << " rms_before=" << before.rms
                << " fluct_before=" << before.fluctuation_rms
                << " dominance=" << dominance
                << " mean_after=" << after.mean
                << " rms_after=" << after.rms
                << " residual_before=" << baseline_check.residual_norm
                << " residual_after=" << recentered_check.residual_norm
                << " accept=" << (accept ? 1 : 0);
            traceLog(oss.str());
        }

        if (accept) {
            return recentered_check;
        }

        x_data = std::move(backup);
        return baseline_check;
    };

    auto undoStageScalingOnSolution = [&](bool blockschur_preparation) {
        if (!blockschur_preparation || stage_scale == 1.0 || solution_stage_scaling_undone) {
            return;
        }
        const Real inv_s = static_cast<Real>(1.0 / stage_scale);
        for (int a = 0; a < lhs.nNo; ++a) {
            for (int c = con_start; c < con_start + con_ncomp; ++c) {
                Ri(c, a) *= inv_s;
            }
        }
        solution_stage_scaling_undone = true;
    };

    fe_fsi_linear_solver::fsils_reset_collective_stats(lhs.commu);
    compareFaceOperatorAgainstFe();
    rebuildPreparedSystem(use_blockschur);
    report.initial_residual_norm = computeOriginalRhsNorm();

    const auto validation_policy = options_.fsils_residual_check_policy;
    const bool require_true_residual_validation = has_native_rank_one_updates;
    auto shouldValidateResidual = [&](bool internal_check_ok, bool recovery_phase) {
        if (require_true_residual_validation) {
            return true;
        }
        switch (validation_policy) {
            case FsilsResidualCheckPolicy::Always:
                return true;
            case FsilsResidualCheckPolicy::RetryOnly:
                return recovery_phase || !internal_check_ok;
            case FsilsResidualCheckPolicy::DebugOnly:
                return recovery_phase || !internal_check_ok || oopTraceEnabled() || lhs.debug_active;
        }
        return recovery_phase || !internal_check_ok;
    };

    auto validateInternalResidual = [&](std::string_view phase) -> FsilsResidualCheckResult {
        FsilsResidualCheckResult result;
        const Real rhs_norm =
            (report.initial_residual_norm > 0.0 && std::isfinite(static_cast<double>(report.initial_residual_norm)))
                ? report.initial_residual_norm
                : static_cast<Real>(std::max(0.0, ls.RI.iNorm));
        result.rhs_norm = rhs_norm;
        result.residual_norm = static_cast<Real>(ls.RI.fNorm);
        result.relative_residual = result.residual_norm / std::max<Real>(rhs_norm, 1e-30);

        const Real target = std::max<Real>(options_.abs_tol, options_.rel_tol * std::max<Real>(rhs_norm, 1e-30));
        const bool finite = std::isfinite(static_cast<double>(ls.RI.iNorm)) &&
                            std::isfinite(static_cast<double>(ls.RI.fNorm)) &&
                            std::isfinite(static_cast<double>(result.relative_residual));
        result.ok = ls.RI.suc && finite && result.residual_norm <= target;
        if (!result.ok) {
            std::ostringstream oss;
            oss << phase << ": internal residual check failed (|r|=" << result.residual_norm
                << ", rel=" << result.relative_residual
                << ", target=" << target
                << ", solver_suc=" << (ls.RI.suc ? 1 : 0) << ")";
            result.detail = oss.str();
        }
        return result;
    };

    auto residualTargetForCheck = [&](const FsilsResidualCheckResult& check) -> Real {
        const Real rhs_norm = std::max<Real>(check.rhs_norm, static_cast<Real>(1e-30));
        return std::max<Real>(options_.abs_tol, options_.rel_tol * rhs_norm);
    };

    auto shouldRetainNearTargetStrictValidationMiss =
        [&](const FsilsResidualCheckResult& check) -> bool {
            if (!(use_blockschur && require_true_residual_validation)) {
                return false;
            }
            if (check.ok) {
                return false;
            }
            if (!std::isfinite(static_cast<double>(check.residual_norm)) ||
                !std::isfinite(static_cast<double>(check.relative_residual))) {
                return false;
            }

            const Real target = residualTargetForCheck(check);
            if (!(target > static_cast<Real>(0.0)) ||
                !std::isfinite(static_cast<double>(target))) {
                return false;
            }

            const Real factor = fsilsStrictValidationNearTargetFactor();
            return check.residual_norm <= factor * target;
        };

    auto runFsilsSolve = [&](bool blockschur_preparation, std::string& error_out) -> bool {
        error_out.clear();
        try {
            const auto& res_current = blockschur_preparation ? res_blockschur : res_original;
            fe_fsi_linear_solver::fsils_solve(lhs, ls, dof, Ri, Val, prec, incL, res_current);
            return true;
        } catch (const std::exception& e) {
            error_out = e.what();
            return false;
        } catch (...) {
            error_out = "unknown exception";
            return false;
        }
    };

    std::string solve_error;
    bool solve_ok = runFsilsSolve(use_blockschur, solve_error);
    FsilsResidualCheckResult initial_check{};
    if (solve_ok) {
        undoStageScalingOnSolution(use_blockschur);
        storeSolveBufferToSolution();
        const std::string phase =
            use_blockschur ? "blockschur" : std::string(solverMethodToString(options_.method));
        initial_check = validateInternalResidual(phase);
        if (shouldValidateResidual(initial_check.ok, /*recovery_phase=*/false)) {
            initial_check = validateOriginalResidual(phase);
            initial_check = maybeRecenterConstraintMeanAndValidate(phase, initial_check);
        }
    } else {
        initial_check.ok = false;
        initial_check.residual_norm = std::numeric_limits<Real>::infinity();
        initial_check.relative_residual = std::numeric_limits<Real>::infinity();
        initial_check.detail = std::string("initial solve threw: ") + solve_error;
    }

    const bool retained_near_target_blockschur_solution =
        shouldRetainNearTargetStrictValidationMiss(initial_check);
    if (retained_near_target_blockschur_solution && oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: retaining near-target BlockSchur solution"
            << " residual=" << initial_check.residual_norm
            << " target=" << residualTargetForCheck(initial_check)
            << " factor=" << fsilsStrictValidationNearTargetFactor();
        traceLog(oss.str());
    }

    int local_fail =
        ((!solve_ok || !initial_check.ok) && !retained_near_target_blockschur_solution) ? 1 : 0;
    int any_fail = local_fail;
    if (lhs.commu.nTasks > 1) {
        fe_fsi_linear_solver::fsils_allreduce(&local_fail, &any_fail, 1, MPI_INT, MPI_LOR, lhs.commu);
    }

    bool used_fallback_gmres = false;
    bool fallback_uses_correction_rhs = false;
    bool skip_strict_retry_after_severe_stall = false;
    std::vector<Real> fallback_base_solution;
    std::string fallback_reason;
    const bool allow_blockschur_gmres_fallback = !use_blockschur;
    if (any_fail != 0 && allow_blockschur_gmres_fallback) {
        fallback_reason = initial_check.detail.empty()
                              ? std::string("initial ") + std::string(solverMethodToString(options_.method)) +
                                    " solve did not satisfy the true residual tolerance"
                              : initial_check.detail;

        if (use_blockschur) {
            preserved_blockschur_stats = ls.blockschur_stats;
            preserved_blockschur_momentum_stats = ls.GM.stats;
            preserved_blockschur_schur_stats = ls.CG.stats;
            preserved_blockschur_attempt =
                preserved_blockschur_stats.outer_iterations > 0 ||
                preserved_blockschur_momentum_stats.solve_calls > 0 ||
                preserved_blockschur_schur_stats.solve_calls > 0;
        }

        if (oopTraceEnabled()) {
            traceLog("FsilsLinearSolver::solve: " + fallback_reason + "; falling back to strict GMRES.");
        }

        int strict_total_iterations =
            std::max(base_gmres_total_iterations, 4 * std::max(1, base_gmres_cfg.sD + 1));
        if (require_true_residual_validation) {
            strict_total_iterations =
                std::max(strict_total_iterations, 40 * std::max(1, base_gmres_cfg.sD + 1));
        }
        const bool promote_restart =
            use_blockschur ||
            gmres_should_promote_restart(options_, initial_check, base_gmres_cfg, ls.RI.itr);
        if (!use_blockschur && !require_true_residual_validation &&
            promote_restart && !gmres_retry_severe_stalls()) {
            skip_strict_retry_after_severe_stall = true;
            if (oopTraceEnabled()) {
                traceLog("FsilsLinearSolver::solve: severe GMRES stall detected; "
                         "skipping strict retry and returning control to caller recovery.");
            }
        }
        used_fallback_gmres = !skip_strict_retry_after_severe_stall;
        const int recovery_sD = promote_restart
                                    ? gmres_recovery_restart_dim(base_gmres_cfg, strict_total_iterations)
                                    : base_gmres_cfg.sD;
        const auto strict_gmres_cfg = make_gmres_launch_config(options_,
                                                               strict_total_iterations,
                                                               recovery_sD);
        const Real original_rhs_norm = std::max<Real>(initial_check.rhs_norm, 1e-30);
        const Real original_target =
            std::max<Real>(options_.abs_tol, options_.rel_tol * original_rhs_norm);
        Real strict_rel_tol =
            (options_.rel_tol > 0.0) ? std::max<Real>(options_.rel_tol * static_cast<Real>(0.1), 1e-14) : 0.0;
        Real strict_abs_tol =
            (options_.abs_tol > 0.0) ? std::max<Real>(options_.abs_tol * static_cast<Real>(0.1), 1e-20) : 0.0;
        if (solve_ok && !skip_strict_retry_after_severe_stall &&
            std::isfinite(static_cast<double>(initial_check.residual_norm)) &&
            std::isfinite(static_cast<double>(initial_check.relative_residual)) &&
            initial_check.residual_norm > 0.0 &&
            original_target > 0.0) {
            const Real correction_rhs_norm = std::max<Real>(initial_check.residual_norm, 1e-30);
            strict_rel_tol = std::clamp<Real>(
                static_cast<Real>(0.5) * original_target / correction_rhs_norm,
                static_cast<Real>(1e-12),
                static_cast<Real>(0.5));
            strict_abs_tol = std::max<Real>(static_cast<Real>(0.5) * original_target, 1e-20);
        }

        if (!skip_strict_retry_after_severe_stall &&
            promote_restart &&
            strict_gmres_cfg.sD > base_gmres_cfg.sD) {
            fallback_reason += "; promoted Krylov dimension from " +
                               std::to_string(base_gmres_cfg.sD) + " to " +
                               std::to_string(strict_gmres_cfg.sD);
            if (oopTraceEnabled()) {
                traceLog("FsilsLinearSolver::solve: severe GMRES stall detected; retrying with krylov_dim=" +
                         std::to_string(strict_gmres_cfg.sD) + ".");
            }
        }

        if (!skip_strict_retry_after_severe_stall) {
            const bool can_use_correction_rhs =
                solve_ok &&
                std::isfinite(static_cast<double>(initial_check.residual_norm)) &&
                std::isfinite(static_cast<double>(initial_check.relative_residual));
            const bool prefer_full_rhs_retry =
                can_use_correction_rhs && has_native_rank_one_updates && has_saddle_point;
            constexpr int max_correction_rounds = 3;
            const int fallback_rounds =
                can_use_correction_rhs
                    ? (max_correction_rounds + (prefer_full_rhs_retry ? 1 : 0))
                    : 1;

            if (can_use_correction_rhs) {
                fallback_base_solution.assign(x_data.begin(), x_data.end());
                fallback_uses_correction_rhs = true;
            }

            for (int round = 0; round < fallback_rounds; ++round) {
                fe_fsi_linear_solver::fsils_ls_create(ls,
                                                      fe_fsi_linear_solver::LS_TYPE_GMRES,
                                                      strict_rel_tol,
                                                      strict_abs_tol,
                                                      strict_gmres_cfg.mItr,
                                                      strict_gmres_cfg.sD);
                ls.RI.exact_convergence = true;
                ls.RI.disable_reorth = false;

                const bool use_full_rhs_retry = prefer_full_rhs_retry && round == 0;
                const bool use_correction_rhs_round =
                    fallback_uses_correction_rhs && !use_full_rhs_retry;

                if (use_correction_rhs_round) {
                    FsilsVector residual_true(shared_layout);
                    Real rhs_norm_unused = 0.0;
                    computeTrueResidualVector(residual_true, rhs_norm_unused);
                    restorePreparedMatrixValues(/*blockschur_preparation=*/false);
                    loadSolveBufferFromVector(residual_true, /*blockschur_preparation=*/false);
                    solution_stage_scaling_undone = false;
                    current_preparation_uses_blockschur = false;
                } else if (round == 0) {
                    rebuildPreparedSystem(/*blockschur_preparation=*/false);
                }

                solve_ok = runFsilsSolve(/*blockschur_preparation=*/false, solve_error);
                if (!fallback_uses_correction_rhs) {
                    break;
                }

                if (!solve_ok) {
                    if (x_data.size() == fallback_base_solution.size()) {
                        std::copy(fallback_base_solution.begin(),
                                  fallback_base_solution.end(),
                                  x_data.begin());
                        solution_stage_scaling_undone = true;
                    }
                    break;
                }

                undoStageScalingOnSolution(/*blockschur_preparation=*/false);
                storeSolveBufferToSolution();
                if (use_full_rhs_retry) {
                    auto full_retry_check = validateOriginalResidual("fallback_gmres_full_retry");
                    full_retry_check = maybeRecenterConstraintMeanAndValidate(
                        "fallback_gmres_full_retry", full_retry_check);
                    fallback_base_solution.assign(x_data.begin(), x_data.end());
                    initial_check = full_retry_check;
                    if (full_retry_check.ok) {
                        break;
                    }
                    if (!std::isfinite(static_cast<double>(full_retry_check.residual_norm))) {
                        break;
                    }
                    continue;
                }
                FE_THROW_IF(x_data.size() != fallback_base_solution.size(), FEException,
                            "FsilsLinearSolver::solve: fallback correction size mismatch");
                for (std::size_t i = 0; i < x_data.size(); ++i) {
                    x_data[i] += fallback_base_solution[i];
                }
                fallback_base_solution.assign(x_data.begin(), x_data.end());
                solution_stage_scaling_undone = true;

                auto refinement_check = validateOriginalResidual("fallback_gmres_refine");
                refinement_check = maybeRecenterConstraintMeanAndValidate(
                    "fallback_gmres_refine", refinement_check);
                if (refinement_check.ok) {
                    break;
                }
                if (!std::isfinite(static_cast<double>(refinement_check.residual_norm)) ||
                    refinement_check.residual_norm >= initial_check.residual_norm * static_cast<Real>(0.95)) {
                    break;
                }
                initial_check = refinement_check;
            }
        }
    }

    FsilsResidualCheckResult final_check{};
    if (skip_strict_retry_after_severe_stall || (use_blockschur && any_fail != 0)) {
        final_check = initial_check;
    } else if (solve_ok) {
        undoStageScalingOnSolution(current_preparation_uses_blockschur);
        if (!(used_fallback_gmres && fallback_uses_correction_rhs)) {
            storeSolveBufferToSolution();
        }
        final_check = validateInternalResidual(used_fallback_gmres ? "fallback_gmres" : "fsils_final");
        if (shouldValidateResidual(final_check.ok, /*recovery_phase=*/used_fallback_gmres)) {
            const std::string phase = used_fallback_gmres ? "fallback_gmres" : "fsils_final";
            final_check = validateOriginalResidual(phase);
            final_check = maybeRecenterConstraintMeanAndValidate(phase, final_check);
        }
    } else {
        final_check.ok = false;
        final_check.rhs_norm = std::max<Real>(report.initial_residual_norm, 0.0);
        final_check.residual_norm = std::numeric_limits<Real>::infinity();
        final_check.relative_residual = std::numeric_limits<Real>::infinity();
        final_check.detail = used_fallback_gmres
                                 ? "fallback_gmres threw: " + solve_error
                                 : "fsils solve threw: " + solve_error;
    }

    int local_final_ok = final_check.ok ? 1 : 0;
    int any_final_ok = local_final_ok;
    if (lhs.commu.nTasks > 1) {
        fe_fsi_linear_solver::fsils_allreduce(&local_final_ok, &any_final_ok, 1, MPI_INT, MPI_LAND, lhs.commu);
    }
    final_check.ok = (any_final_ok != 0);
    if (!final_check.ok && final_check.detail.empty()) {
        final_check.detail = "true residual check failed on another rank";
        final_check.residual_norm = std::numeric_limits<Real>::infinity();
        final_check.relative_residual = std::numeric_limits<Real>::infinity();
    }

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
    lhs.native_face_rank_one_count = 0;
    lhs.reduced_updates.clear();
    lhs.grouped_bordered_field_couplings.clear();

    if (solve_ok && oopTraceEnabled()) {
        logInternalBlockSolutionStats("returned");
        logReturnedOperatorBreakdown("pre_report", *x);

        FsilsVector matvec_only(shared_layout);
        A->mult(*x, matvec_only);

        FsilsVector matvec_full(shared_layout);
        matvec_full.copyFrom(matvec_only);
        addRankOneUpdatesToProduct(rank_one_updates_, *x, matvec_full, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_, *x, matvec_full, lhs.commu);

        FsilsVector rhs_check(shared_layout);
        rhs_check.copyFrom(*b);
        rhs_check.accumulateOverlap();

        FsilsVector x_check(shared_layout);
        x_check.copyFrom(*x);
        x_check.updateGhosts();

        FsilsVector diff_only(shared_layout);
        A->mult(x_check, diff_only);
        auto diff_only_span = diff_only.localSpan();
        const auto rhs_span = rhs_check.localSpan();
        for (std::size_t i = 0; i < diff_only_span.size(); ++i) {
            diff_only_span[i] -= rhs_span[i];
        }

        FsilsVector diff_full(shared_layout);
        diff_full.copyFrom(diff_only);
        addRankOneUpdatesToProduct(rank_one_updates_, x_check, diff_full, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_, x_check, diff_full, lhs.commu);
        auto diff_full_span = diff_full.localSpan();
        for (std::size_t i = 0; i < diff_full_span.size(); ++i) {
            diff_full_span[i] -= rhs_span[i];
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: post-return operator check"
            << " |Jx|=" << diff_only.norm()
            << " |(J+R)x|=" << diff_full.norm()
            << " |Jx-r|=" << diff_only.norm()
            << " |(J+R)x-r|=" << diff_full.norm();
        traceLog(oss.str());
    }

    report.iterations = ls.RI.itr;
    report.final_residual_norm = final_check.residual_norm;
    report.relative_residual = final_check.relative_residual;
    report.converged = final_check.ok;
    report.message = used_fallback_gmres
                         ? "fsils (fallback gmres)"
                         : (use_blockschur ? "fsils (blockschur)" : "fsils");
    if (used_fallback_gmres && !fallback_reason.empty()) {
        report.message = "fsils (fallback gmres: " + fallback_reason + ")";
    } else if (!final_check.ok && !final_check.detail.empty()) {
        report.message = "fsils (" + final_check.detail + ")";
    }

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
        const Real rel_tol = std::max<Real>(options_.rel_tol, 0.0);
        const Real abs_tol = std::max<Real>(options_.abs_tol, 0.0);
        const Real target = std::max(abs_tol, rel_tol * report.initial_residual_norm);
        const bool meets_target = report.final_residual_norm <= target;
        if (!report.converged && meets_target) {
            report.converged = true;
            report.message = used_fallback_gmres
                                 ? "fsils (fallback gmres)"
                                 : (use_blockschur ? "fsils (blockschur)" : "fsils");
        } else if (!report.converged) {
            const std::string rel_msg = "fsils (not converged; itr=" + std::to_string(report.iterations) +
                                        ", rel=" + std::to_string(report.relative_residual) + ")";
            if (!final_check.ok && !final_check.detail.empty()) {
                report.message = "fsils (" + final_check.detail + ")";
            } else {
                report.message = used_fallback_gmres
                    ? "fsils (fallback gmres; not converged; itr=" + std::to_string(report.iterations) +
                          ", rel=" + std::to_string(report.relative_residual) + ")"
                    : rel_msg;
            }
        }
    }

    // Post-solve nullspace projection: x = x - Σ_i (z_i · x) z_i
    // This removes any nullspace drift from the iterative solve.
    const bool applied_nullspace_projection = !nullspace_basis_.empty() && x != nullptr;
    const Real x_norm_before_nullspace =
        (x != nullptr) ? x->norm() : static_cast<Real>(0.0);
    if (applied_nullspace_projection) {
        auto x_span = x->localSpan();
        const auto n = x_span.size();

        for (const auto& z : nullspace_basis_) {
            if (z.size() != n) continue;

            // Compute local dot product z · x
            double local_dot = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                local_dot += z[i] * static_cast<double>(x_span[i]);
            }

            // MPI_Allreduce for distributed dot product
            double global_dot = local_dot;
#if FE_HAS_MPI
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized && lhs.commu.nTasks > 1) {
                fe_fsi_linear_solver::fsils_allreduce_sum(
                    &local_dot, &global_dot, 1, MPI_DOUBLE, lhs.commu);
            }
#endif

            // x = x - (z · x) * z
            for (std::size_t i = 0; i < n; ++i) {
                x_span[i] -= static_cast<Real>(global_dot * z[i]);
            }
        }
    }

    if (solve_ok && oopTraceEnabled()) {
        logReturnedOperatorBreakdown("final", *x);

        FsilsVector rhs_check(shared_layout);
        rhs_check.copyFrom(*b);
        rhs_check.accumulateOverlap();

        FsilsVector x_check(shared_layout);
        x_check.copyFrom(*x);
        x_check.updateGhosts();

        FsilsVector matvec_full(shared_layout);
        A->mult(x_check, matvec_full);
        addRankOneUpdatesToProduct(rank_one_updates_, x_check, matvec_full, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_, x_check, matvec_full, lhs.commu);

        FsilsVector diff_full(shared_layout);
        diff_full.copyFrom(matvec_full);
        auto diff_full_span = diff_full.localSpan();
        const auto rhs_span = rhs_check.localSpan();
        for (std::size_t i = 0; i < diff_full_span.size(); ++i) {
            diff_full_span[i] -= rhs_span[i];
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: final returned operator check"
            << " basis=" << nullspace_basis_.size()
            << " projected=" << (applied_nullspace_projection ? 1 : 0)
            << " |x_before_proj|=" << x_norm_before_nullspace
            << " |x_after_proj|=" << x->norm()
            << " |(J+R)x-r|=" << diff_full.norm();
        traceLog(oss.str());
    }

    report.setup_time_seconds = rhs_prepare_time_seconds;
    report.validation_time_seconds = validation_time_seconds;
    report.collective_time_seconds = lhs.commu.collective_stats.allreduce_time;
    report.collective_calls = lhs.commu.collective_stats.allreduce_calls;
    report.collective_words = lhs.commu.collective_stats.allreduce_words;
    const auto& blockschur_stats_for_report =
        (used_fallback_gmres && preserved_blockschur_attempt) ? preserved_blockschur_stats
                                                              : ls.blockschur_stats;
    const auto& blockschur_momentum_stats_for_report =
        (used_fallback_gmres && preserved_blockschur_attempt) ? preserved_blockschur_momentum_stats
                                                              : ls.GM.stats;
    const auto& blockschur_schur_stats_for_report =
        (used_fallback_gmres && preserved_blockschur_attempt) ? preserved_blockschur_schur_stats
                                                              : ls.CG.stats;

    report.blockschur_outer_iterations = blockschur_stats_for_report.outer_iterations;
    report.blockschur_collective_calls_max_per_outer =
        blockschur_stats_for_report.collective_calls_max_per_outer;
    report.blockschur_collective_time_max_per_outer =
        blockschur_stats_for_report.collective_time_max_per_outer;
    report.blockschur_momentum_solve_calls = blockschur_momentum_stats_for_report.solve_calls;
    report.blockschur_momentum_iterations = blockschur_momentum_stats_for_report.iterations_total;
    report.blockschur_momentum_restart_cycles =
        blockschur_momentum_stats_for_report.restart_cycles_total;
    report.blockschur_momentum_solve_time_seconds = blockschur_momentum_stats_for_report.solve_time;
    report.blockschur_momentum_collective_calls = blockschur_momentum_stats_for_report.collective_calls;
    report.blockschur_momentum_collective_words = blockschur_momentum_stats_for_report.collective_words;
    report.blockschur_momentum_collective_time_seconds =
        blockschur_momentum_stats_for_report.collective_time;
    report.blockschur_schur_solve_calls = blockschur_schur_stats_for_report.solve_calls;
    report.blockschur_schur_iterations = blockschur_schur_stats_for_report.iterations_total;
    report.blockschur_schur_setup_time_seconds = blockschur_schur_stats_for_report.setup_time;
    report.blockschur_schur_solve_time_seconds = blockschur_schur_stats_for_report.solve_time;
    report.blockschur_schur_collective_calls = blockschur_schur_stats_for_report.collective_calls;
    report.blockschur_schur_collective_words = blockschur_schur_stats_for_report.collective_words;
    report.blockschur_schur_collective_time_seconds = blockschur_schur_stats_for_report.collective_time;

    if (lhs.commu.task == 0) {
        std::fprintf(stderr,
                     "\n=== FSILS BACKEND METRICS (rank 0) ===\n"
                     "  RHS/overlap prep:     %10.6f s\n"
                     "  Residual validation:  %10.6f s\n"
                     "  MPI_Allreduce calls:  %10llu\n"
                     "  MPI_Allreduce words:  %10llu\n"
                     "  MPI_Allreduce time:   %10.6f s\n"
                     "=======================================\n",
                     report.setup_time_seconds,
                     report.validation_time_seconds,
                     static_cast<unsigned long long>(report.collective_calls),
                     static_cast<unsigned long long>(report.collective_words),
                     report.collective_time_seconds);
        if (use_blockschur) {
            const double calls_per_outer =
                (report.blockschur_outer_iterations > 0)
                    ? static_cast<double>(report.collective_calls) /
                          static_cast<double>(report.blockschur_outer_iterations)
                    : 0.0;
            const double calls_per_restart =
                (report.blockschur_momentum_restart_cycles > 0)
                    ? static_cast<double>(report.blockschur_momentum_collective_calls) /
                          static_cast<double>(report.blockschur_momentum_restart_cycles)
                    : 0.0;
            std::fprintf(stderr,
                         "  BlockSchur outer iters: %8d\n"
                         "  Calls / outer iter:     %10.3f\n"
                         "  Max calls / outer:      %10llu\n"
                         "  Momentum solves:        %8d  iters=%d  restarts=%d\n"
                         "  Momentum solve time:    %10.6f s\n"
                         "  Momentum allreduces:    %10llu  words=%llu  time=%10.6f s\n"
                         "  Calls / GMRES restart:  %10.3f\n"
                         "  Schur solves:           %8d  iters=%d\n"
                         "  Schur setup time:       %10.6f s\n"
                         "  Schur solve time:       %10.6f s\n"
                         "  Schur allreduces:       %10llu  words=%llu  time=%10.6f s\n"
                         "=======================================\n",
                         report.blockschur_outer_iterations,
                         calls_per_outer,
                         static_cast<unsigned long long>(report.blockschur_collective_calls_max_per_outer),
                         report.blockschur_momentum_solve_calls,
                         report.blockschur_momentum_iterations,
                         report.blockschur_momentum_restart_cycles,
                         report.blockschur_momentum_solve_time_seconds,
                         static_cast<unsigned long long>(report.blockschur_momentum_collective_calls),
                         static_cast<unsigned long long>(report.blockschur_momentum_collective_words),
                         report.blockschur_momentum_collective_time_seconds,
                         calls_per_restart,
                         report.blockschur_schur_solve_calls,
                         report.blockschur_schur_iterations,
                         report.blockschur_schur_setup_time_seconds,
                         report.blockschur_schur_solve_time_seconds,
                         static_cast<unsigned long long>(report.blockschur_schur_collective_calls),
                         static_cast<unsigned long long>(report.blockschur_schur_collective_words),
                         report.blockschur_schur_collective_time_seconds);
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

void FsilsLinearSolver::setNullspaceBasis(std::span<const std::vector<double>> basis)
{
    nullspace_basis_.clear();
    nullspace_basis_.reserve(basis.size());
    for (const auto& vec : basis) {
        nullspace_basis_.push_back(vec);
    }
    if (oopTraceEnabled()) {
        traceLog("FsilsLinearSolver::setNullspaceBasis: modes=" +
                 std::to_string(nullspace_basis_.size()));
    }
}

} // namespace backends
} // namespace FE
} // namespace svmp
