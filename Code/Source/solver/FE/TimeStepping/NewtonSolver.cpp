/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/NewtonSolver.h"

#include "Backends/Interfaces/BackendFactory.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/GaugeDiagnostics.h"
#include "Constraints/GaugeRegistry.h"
#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Dofs/DofIndexSet.h"
#include "Dofs/EntityDofMap.h"
#include "Systems/AuxiliaryOperatorRegistry.h"
#include "Systems/AuxiliaryStateManager.h"
#include "Systems/SystemsExceptions.h"

#if defined(FE_HAS_FSILS)
#  include "Backends/FSILS/FsilsMatrix.h"
#  include "Backends/FSILS/FsilsVector.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace timestepping {

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

[[nodiscard]] int mpiRank() noexcept
{
#if FE_HAS_MPI
    int rank = 0;
    (void)MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

[[nodiscard]] bool jacobianCheckEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK");
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

[[nodiscard]] int jacobianCheckNewtonIteration() noexcept
{
    static const int iter = [] {
        const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK_IT");
        if (env == nullptr) {
            return 0;
        }
        char* end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end == env) {
            return 0;
        }
        if (v < 0) {
            return 0;
        }
        if (v > std::numeric_limits<int>::max()) {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(v);
    }();
    return iter;
}

[[nodiscard]] double jacobianCheckRelativeStep() noexcept
{
    static const double step = [] {
        const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK_STEP");
        if (env == nullptr) {
            return 1e-7;
        }
        char* end = nullptr;
        const double v = std::strtod(env, &end);
        if (end == env) {
            return 1e-7;
        }
        if (!(v > 0.0) || !std::isfinite(v)) {
            return 1e-7;
        }
        return v;
    }();
    return step;
}

void axpy(backends::GenericVector& y, Real alpha, const backends::GenericVector& x)
{
    auto ys = y.localSpan();
    auto xs = x.localSpan();
    FE_CHECK_ARG(ys.size() == xs.size(), "NewtonSolver: axpy size mismatch");
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += alpha * xs[i];
    }
}

void copyVector(backends::GenericVector& dst, const backends::GenericVector& src)
{
    auto d = dst.localSpan();
    auto s = src.localSpan();
    FE_CHECK_ARG(d.size() == s.size(), "NewtonSolver: copyVector size mismatch");
    std::copy(s.begin(), s.end(), d.begin());
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

double auxiliaryResidualNormForConvergence(const systems::FESystem::BorderedCouplingData& bordered)
{
    if (!bordered.active || bordered.g.empty()) {
        return 0.0;
    }

    long double local_sq = 0.0L;
    for (const auto v : bordered.g) {
        local_sq += static_cast<long double>(v) * static_cast<long double>(v);
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        long double global_sq = 0.0L;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        local_sq = global_sq;
    }
#endif

    return std::sqrt(static_cast<double>(local_sq));
}

double borderedResidualNormForConvergence(const backends::GenericVector& r,
                                          backends::GenericVector& scratch,
                                          const systems::FESystem::BorderedCouplingData& bordered)
{
    const double pde_norm = residualNormForConvergence(r, scratch);
    const double aux_norm = auxiliaryResidualNormForConvergence(bordered);
    return std::hypot(pde_norm, aux_norm);
}

[[nodiscard]] std::pair<double, double> borderedResidualNormComponentsForConvergence(
    const backends::GenericVector& r,
    backends::GenericVector& scratch,
    const systems::FESystem::BorderedCouplingData& bordered)
{
    return {
        residualNormForConvergence(r, scratch),
        auxiliaryResidualNormForConvergence(bordered)
    };
}

void zeroVectorEntries(std::span<const GlobalIndex> dofs, backends::GenericVector& vec)
{
    if (dofs.empty()) {
        return;
    }
    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: zeroVectorEntries view");
    view->beginAssemblyPhase();
    view->zeroVectorEntries(dofs);
    view->finalizeAssembly();
}

void accumulateOverlapIfNeeded(backends::GenericVector& vec)
{
#if defined(FE_HAS_FSILS)
    if (vec.backendKind() != backends::BackendKind::FSILS) {
        return;
    }
    auto* fs = dynamic_cast<backends::FsilsVector*>(&vec);
    if (fs == nullptr) {
        return;
    }
    fs->accumulateOverlap();
#else
    (void)vec;
#endif
}

struct FsilsMatrixSnapshot {
    std::vector<Real> values{};

    [[nodiscard]] bool valid() const noexcept { return !values.empty(); }
};

struct SolverOptionsGuard {
    backends::LinearSolver& linear;
    backends::SolverOptions saved;
    ~SolverOptionsGuard() noexcept
    {
        try {
            linear.setOptions(saved);
        } catch (...) {
        }
    }
};

[[nodiscard]] backends::SolverOptions makeBorderedSolveOptions(const backends::SolverOptions& base)
{
    backends::SolverOptions opts = base;

    const Real target_rel =
        (base.rel_tol > Real(0.0) && std::isfinite(static_cast<double>(base.rel_tol)))
            ? std::min(static_cast<Real>(1e-8), static_cast<Real>(base.rel_tol * Real(1e-2)))
            : static_cast<Real>(1e-8);
    opts.rel_tol = std::max(target_rel, static_cast<Real>(1e-12));

    const Real target_abs =
        (base.abs_tol > Real(0.0) && std::isfinite(static_cast<double>(base.abs_tol)))
            ? std::min(base.abs_tol, static_cast<Real>(1e-12))
            : static_cast<Real>(1e-12);
    opts.abs_tol = std::max(target_abs, static_cast<Real>(1e-16));

    // Strongly coupled bordered solves are much less tolerant of loose inner
    // solves than the legacy PDE-only path. Keep the cap finite for FSILS, but
    // allow materially more Krylov work before declaring failure.
    opts.max_iter = std::max(base.max_iter, 200);
    opts.max_iter = std::min(opts.max_iter, 200);

    if (base.method == backends::SolverMethod::BlockSchur) {
        const int target_inner_max_iter = std::max(base.max_iter, 200);
        opts.fsils_blockschur_gm_max_iter =
            std::max(base.fsils_blockschur_gm_max_iter.value_or(0), target_inner_max_iter);
        opts.fsils_blockschur_cg_max_iter =
            std::max(base.fsils_blockschur_cg_max_iter.value_or(0), target_inner_max_iter);

        const Real gm_target_rel = base.fsils_blockschur_gm_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_gm_rel_tol, opts.rel_tol)
            : opts.rel_tol;
        const Real cg_target_rel = base.fsils_blockschur_cg_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_cg_rel_tol, opts.rel_tol)
            : opts.rel_tol;
        opts.fsils_blockschur_gm_rel_tol = gm_target_rel;
        opts.fsils_blockschur_cg_rel_tol = cg_target_rel;
    }

    return opts;
}

[[nodiscard]] FsilsMatrixSnapshot captureFsilsMatrixSnapshot(const backends::GenericMatrix& A)
{
#if defined(FE_HAS_FSILS)
    const auto* fs = dynamic_cast<const backends::FsilsMatrix*>(&A);
    if (!fs) {
        return {};
    }
    const auto nnz = fs->fsilsNnz();
    const auto dof = fs->fsilsDof();
    if (nnz <= 0 || dof <= 0) {
        return {};
    }

    FsilsMatrixSnapshot snap;
    const auto count = static_cast<std::size_t>(nnz) *
                       static_cast<std::size_t>(dof) *
                       static_cast<std::size_t>(dof);
    snap.values.resize(count);
    std::copy(fs->fsilsValuesPtr(), fs->fsilsValuesPtr() + count, snap.values.begin());
    return snap;
#else
    (void)A;
    return {};
#endif
}

void restoreFsilsMatrixSnapshot(backends::GenericMatrix& A, const FsilsMatrixSnapshot& snap)
{
#if defined(FE_HAS_FSILS)
    if (!snap.valid()) {
        return;
    }
    auto* fs = dynamic_cast<backends::FsilsMatrix*>(&A);
    if (!fs) {
        return;
    }
    const auto nnz = fs->fsilsNnz();
    const auto dof = fs->fsilsDof();
    if (nnz <= 0 || dof <= 0) {
        return;
    }
    const auto count = static_cast<std::size_t>(nnz) *
                       static_cast<std::size_t>(dof) *
                       static_cast<std::size_t>(dof);
    FE_CHECK_ARG(snap.values.size() == count,
                 "NewtonSolver: FSILS matrix snapshot size mismatch");
    std::copy(snap.values.begin(), snap.values.end(), fs->fsilsValuesPtr());
#else
    (void)A;
    (void)snap;
#endif
}

[[nodiscard]] bool solveDenseLinearSystem(std::vector<Real>& A,
                                          std::vector<Real>& b,
                                          Real pivot_tol = static_cast<Real>(1e-20))
{
    const auto n = b.size();
    if (A.size() != n * n) {
        return false;
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        Real pivot_abs = std::abs(A[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real cand = std::abs(A[i * n + k]);
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot = i;
            }
        }
        if (!(pivot_abs > pivot_tol)) {
            return false;
        }
        if (pivot != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[pivot * n + j]);
            }
            std::swap(b[k], b[pivot]);
        }

        const Real diag = A[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real factor = A[i * n + k] / diag;
            if (std::abs(factor) <= pivot_tol) {
                continue;
            }
            A[i * n + k] = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
        Real sum = b[static_cast<std::size_t>(i)];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
            sum -= A[static_cast<std::size_t>(i) * n + j] * b[j];
        }
        const Real diag = A[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(i)];
        if (!(std::abs(diag) > pivot_tol)) {
            return false;
        }
        b[static_cast<std::size_t>(i)] = sum / diag;
    }

    return true;
}

void applyAuxiliaryDelta(systems::FESystem& system,
                         const systems::FESystem::BorderedCouplingData& bc,
                         std::span<const Real> dx,
                         Real alpha)
{
    if (!bc.active || dx.empty() || !(alpha != Real(0.0))) {
        return;
    }

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    FE_CHECK_NOT_NULL(mgr, "NewtonSolver: auxiliary state manager");

    std::size_t offset = 0;
    for (const auto& blk_info : bc.aux_blocks) {
        auto& blk = mgr->getBlock(blk_info.name);
        auto work = blk.work();
        const auto block_dim = static_cast<std::size_t>(blk_info.dim);
        FE_THROW_IF(offset + block_dim > dx.size(), systems::InvalidStateException,
                    "NewtonSolver: auxiliary bordered update exceeds dx size");
        FE_THROW_IF(work.size() != block_dim, systems::InvalidStateException,
                    "NewtonSolver: auxiliary bordered update size mismatch for block '" +
                        blk_info.name + "'");

        for (std::size_t i = 0; i < block_dim; ++i) {
            work[i] -= alpha * dx[offset + i];
        }
        offset += block_dim;
    }

    FE_THROW_IF(offset != dx.size(), systems::InvalidStateException,
                "NewtonSolver: auxiliary bordered update did not consume all dx entries");
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

    FE_THROW_IF(options_.jacobian_rebuild_period <= 0, InvalidArgumentException,
                "NewtonSolver: jacobian_rebuild_period must be >= 1");
    if (options_.scale_dt_increments) {
        FE_THROW_IF(!std::isfinite(options_.dt_increment_scale), InvalidArgumentException,
                    "NewtonSolver: dt_increment_scale must be finite");
        FE_THROW_IF(options_.dt_increment_scale < 0.0, InvalidArgumentException,
                    "NewtonSolver: dt_increment_scale must be >= 0");
    }

    if (options_.use_line_search) {
        FE_THROW_IF(options_.line_search_max_iterations <= 0, InvalidArgumentException,
                    "NewtonSolver: line_search_max_iterations must be > 0 when line search is enabled");
        FE_THROW_IF(!(options_.line_search_alpha_min > 0.0) || options_.line_search_alpha_min > 1.0 ||
                        !std::isfinite(options_.line_search_alpha_min),
                    InvalidArgumentException,
                    "NewtonSolver: line_search_alpha_min must be finite and in (0,1]");
        FE_THROW_IF(!(options_.line_search_shrink > 0.0) || options_.line_search_shrink >= 1.0 ||
                        !std::isfinite(options_.line_search_shrink),
                    InvalidArgumentException,
                    "NewtonSolver: line_search_shrink must be finite and in (0,1)");
        FE_THROW_IF(!(options_.line_search_c1 > 0.0) || options_.line_search_c1 >= 1.0 ||
                        !std::isfinite(options_.line_search_c1),
                    InvalidArgumentException,
                    "NewtonSolver: line_search_c1 must be finite and in (0,1)");
    }

    if (options_.pseudo_transient.enabled) {
        FE_THROW_IF(options_.pseudo_transient.gamma_initial < 0.0 ||
                        !std::isfinite(options_.pseudo_transient.gamma_initial),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_initial must be finite and >= 0");
        FE_THROW_IF(!(options_.pseudo_transient.gamma_growth > 1.0) ||
                        !std::isfinite(options_.pseudo_transient.gamma_growth),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_growth must be finite and > 1");
        FE_THROW_IF(options_.pseudo_transient.gamma_max < 0.0 ||
                        !std::isfinite(options_.pseudo_transient.gamma_max),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_max must be finite and >= 0");
        FE_THROW_IF(options_.pseudo_transient.gamma_drop_tolerance < 0.0 ||
                        !std::isfinite(options_.pseudo_transient.gamma_drop_tolerance),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_drop_tolerance must be finite and >= 0");
        FE_THROW_IF(options_.pseudo_transient.max_linear_retries <= 0,
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.max_linear_retries must be > 0");
    }
}

systems::SystemStateView NewtonSolver::makeStateView(const TimeHistory& history, double solve_time) const
{
    systems::SystemStateView state;
    state.time = solve_time;
    state.dt = history.dt();
    const double stage_dt = solve_time - history.time();
    state.effective_dt =
        (std::isfinite(stage_dt) && stage_dt > 0.0) ? stage_dt : history.dt();
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
    workspace.u_backup = factory.createVector(n_dofs);
    workspace.residual_scratch = factory.createVector(n_dofs);
    workspace.residual_base = factory.createVector(n_dofs);
    workspace.ptc_mass_lumped.reset();
    workspace.dt_field_dofs.clear();

    FE_CHECK_NOT_NULL(workspace.jacobian.get(), "NewtonSolver workspace.jacobian");
    FE_CHECK_NOT_NULL(workspace.residual.get(), "NewtonSolver workspace.residual");
    FE_CHECK_NOT_NULL(workspace.delta.get(), "NewtonSolver workspace.delta");
    FE_CHECK_NOT_NULL(workspace.u_backup.get(), "NewtonSolver workspace.u_backup");
    FE_CHECK_NOT_NULL(workspace.residual_scratch.get(), "NewtonSolver workspace.residual_scratch");
    FE_CHECK_NOT_NULL(workspace.residual_base.get(), "NewtonSolver workspace.residual_base");

    if (options_.pseudo_transient.enabled) {
        workspace.ptc_mass_lumped = factory.createVector(n_dofs);
        FE_CHECK_NOT_NULL(workspace.ptc_mass_lumped.get(), "NewtonSolver workspace.ptc_mass_lumped");
    }

    if (options_.scale_dt_increments) {
        const auto dt_fields = system.timeDerivativeFields();
        if (!dt_fields.empty()) {
            const auto& fmap = system.fieldMap();
            for (const auto fid : dt_fields) {
                if (fid < 0) {
                    continue;
                }
                const auto idx = static_cast<std::size_t>(fid);
                if (idx >= fmap.numFields()) {
                    continue;
                }
                const auto range = fmap.getFieldDofRange(idx);
                for (GlobalIndex d = range.first; d < range.second; ++d) {
                    workspace.dt_field_dofs.push_back(d);
                }
            }
            std::sort(workspace.dt_field_dofs.begin(), workspace.dt_field_dofs.end());
            workspace.dt_field_dofs.erase(
                std::unique(workspace.dt_field_dofs.begin(), workspace.dt_field_dofs.end()),
                workspace.dt_field_dofs.end());
        }
    }

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "NewtonSolver::allocateWorkspace: backend=" << backends::backendKindToString(factory.backendKind())
            << " ndofs=" << n_dofs << " jacobian_op='" << options_.jacobian_op << "'"
            << " residual_op='" << options_.residual_op << "'"
            << " dist_sparsity=" << ((dist != nullptr && factory.backendKind() != backends::BackendKind::Eigen) ? "yes" : "no")
            << " dt_field_dofs=" << workspace.dt_field_dofs.size();
        traceLog(oss.str());
    }
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
    auto& u_backup = *workspace.u_backup;
    auto& residual_scratch = *workspace.residual_scratch;
    auto& residual_base = *workspace.residual_base;

    NewtonReport report;

    const auto& sys = transient.system();
    const auto& constraints = sys.constraints();
    const int temporal_order = transient.system().temporalOrder();

    history.updateGhosts();

    struct NewtonStateWithContext {
        systems::SystemStateView view{};
        std::optional<assembly::TimeIntegrationContext> time_ctx{};
    };

    auto makeNewtonState = [&](const TimeHistory& hist, double time) {
        NewtonStateWithContext out;
        out.view = makeStateView(hist, time);
        if (temporal_order > 0) {
            out.time_ctx = transient.integrator().buildContext(temporal_order, out.view);
            out.view.time_integration = &(*out.time_ctx);
        }
        return out;
    };

    auto base_state_holder = makeNewtonState(history, solve_time);
    const auto& base_state = base_state_holder.view;
    FE_THROW_IF(!(base_state.dt > 0.0), InvalidArgumentException, "NewtonSolver: dt must be > 0");
    FE_THROW_IF(!std::isfinite(base_state.time), InvalidArgumentException, "NewtonSolver: solve_time must be finite");

    // Ensure time-dependent constraints (Dirichlet, etc.) are evaluated at the actual solve time.
    // This is required for multi-stage schemes (e.g., generalized-α) where the nonlinear solve
    // occurs at a stage time t_{n+α_f}, not necessarily at t_{n+1}.
    transient.system().updateConstraints(solve_time, base_state.dt);

    std::optional<assembly::TimeIntegrationContext> dt_scale_ctx;
    if (options_.scale_dt_increments && !(options_.dt_increment_scale > 0.0)) {
        const int max_order = transient.system().temporalOrder();
        if (max_order > 0) {
            dt_scale_ctx = transient.integrator().buildContext(max_order, base_state);
        }
    }

    const int max_it = options_.max_iterations;

    if (oopTraceEnabled()) {
        const auto& lopts = linear.getOptions();
        std::ostringstream oss;
        oss << "NewtonSolver::solveStep: time=" << solve_time << " dt=" << base_state.dt
            << " max_it=" << max_it
            << " abs_tol=" << options_.abs_tolerance << " rel_tol=" << options_.rel_tolerance
            << " step_tol=" << options_.step_tolerance
            << " residual_op='" << options_.residual_op << "' jacobian_op='" << options_.jacobian_op << "'"
            << " linear_backend=" << backends::backendKindToString(linear.backendKind())
            << " linear(method=" << backends::solverMethodToString(lopts.method)
            << ", pc=" << backends::preconditionerToString(lopts.preconditioner)
            << ", max_iter=" << lopts.max_iter
            << ", rel_tol=" << lopts.rel_tol
            << ", abs_tol=" << lopts.abs_tol << ")";
        traceLog(oss.str());
    }

    const bool same_op = (options_.residual_op == options_.jacobian_op);
    bool has_monolithic_auxiliary_unknowns = false;
    if (const auto* aux_registry = transient.system().auxiliaryOperatorRegistryIfPresent();
        aux_registry && aux_registry->isLayoutFinalized()) {
        has_monolithic_auxiliary_unknowns =
            aux_registry->auxiliaryLayout().total_aux_unknowns > 0;
    }

    std::vector<GlobalIndex> constrained_dofs;
    std::vector<GlobalIndex> dirichlet_dofs;
    if (!constraints.empty()) {
        constrained_dofs.reserve(constraints.numConstraints());
        dirichlet_dofs.reserve(constraints.numConstraints());
        constraints.forEach([&constrained_dofs](const constraints::AffineConstraints::ConstraintView& cv) {
            if (cv.slave_dof >= 0) {
                constrained_dofs.push_back(cv.slave_dof);
            }
        });
        constraints.forEach([&dirichlet_dofs](const constraints::AffineConstraints::ConstraintView& cv) {
            if (cv.slave_dof >= 0 && cv.isDirichlet()) {
                dirichlet_dofs.push_back(cv.slave_dof);
            }
        });

        std::sort(constrained_dofs.begin(), constrained_dofs.end());
        constrained_dofs.erase(std::unique(constrained_dofs.begin(), constrained_dofs.end()),
                               constrained_dofs.end());

        std::sort(dirichlet_dofs.begin(), dirichlet_dofs.end());
        dirichlet_dofs.erase(std::unique(dirichlet_dofs.begin(), dirichlet_dofs.end()),
                             dirichlet_dofs.end());
    }
    linear.setDirichletDofs(dirichlet_dofs);

    auto zeroConstrainedResidualEntries = [&]() {
        if (constrained_dofs.empty()) {
            return;
        }
        auto r_zero = r.createAssemblyView();
        FE_CHECK_NOT_NULL(r_zero.get(), "NewtonSolver: residual zeroing view");
        r_zero->beginAssemblyPhase();
        r_zero->zeroVectorEntries(constrained_dofs);
        r_zero->finalizeAssembly();
    };

    auto applyResidualAdditionAndConstraints = [&]() {
        if (residual_addition != nullptr) {
            axpy(r, static_cast<Real>(1.0), *residual_addition);
        }
        zeroConstrainedResidualEntries();
    };

    auto reapplyConstrainedJacobianRows = [&]() {
        if (constrained_dofs.empty()) {
            return;
        }
        auto J_zero = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_zero.get(), "NewtonSolver: Jacobian constrained-row view");
        J_zero->beginAssemblyPhase();
        J_zero->zeroRows(constrained_dofs, /*set_diagonal=*/true);
        J_zero->finalizeAssembly();
    };

    auto computeResidualNorm = [&]() -> double {
        return borderedResidualNormForConvergence(
            r, residual_scratch, transient.system().borderedCoupling());
    };

    auto traceResidualComponents = [&](const char* phase) {
        if (!oopTraceEnabled()) {
            return;
        }
        const auto [pde_norm, aux_norm] = borderedResidualNormComponentsForConvergence(
            r, residual_scratch, transient.system().borderedCoupling());
        std::ostringstream oss;
        oss << "NewtonSolver: residual components";
        if (phase != nullptr) {
            oss << " phase='" << phase << "'";
        }
        oss << " pde=" << pde_norm
            << " aux=" << aux_norm
            << " combined=" << std::hypot(pde_norm, aux_norm);
        traceLog(oss.str());
    };

    const bool ptc_enabled = options_.pseudo_transient.enabled;
    std::vector<GlobalIndex> ptc_owned_dofs;
    if (ptc_enabled && workspace.ptc_mass_lumped != nullptr) {
        const auto dt_fields = sys.timeDerivativeFields(options_.jacobian_op);
        if (!dt_fields.empty()) {
            dofs::IndexSet dt_dofs_all;
            const auto& fmap = sys.fieldMap();
            for (const auto fid : dt_fields) {
                if (fid < 0) {
                    continue;
                }
                const auto idx = static_cast<std::size_t>(fid);
                if (idx >= fmap.numFields()) {
                    continue;
                }
                const auto range = fmap.getFieldDofRange(idx);
                dt_dofs_all = dt_dofs_all.unionWith(dofs::IndexSet(range.first, range.second));
            }
            const auto& owned = sys.dofHandler().getPartition().locallyOwned();
            ptc_owned_dofs = dt_dofs_all.intersectionWith(owned).toVector();
        }
    }

    const bool ptc_can_run = ptc_enabled && (workspace.ptc_mass_lumped != nullptr) && !ptc_owned_dofs.empty();
    bool ptc_mass_ready = false;
    double ptc_gamma = 0.0;
    double ptc_gamma_applied = 0.0;
    double ptc_prev_residual_norm = std::numeric_limits<double>::quiet_NaN();

    systems::OperatorTag residual_op_used = options_.residual_op;

    auto assembleResidualOnly = [&](const systems::SystemStateView& state, const char* phase) -> double {
        residual_op_used = options_.residual_op;
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");

        if (oopTraceEnabled()) {
            std::string msg = "NewtonSolver: beginTimeStep() + assemble (vector) op='" + options_.residual_op + "'";
            if (phase != nullptr) {
                msg += " phase='";
                msg += phase;
                msg += "'";
            }
            traceLog(msg);
        }

        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, nullptr, r_view.get());
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: residual assembly failed: " + ar.error_message);

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=0 want_vector=1"
                << " ok=" << ar.success
                << " elems=" << ar.elements_assembled
                << " vec_ins=" << ar.vector_entries_inserted
                << " time=" << ar.elapsed_time_seconds;
            if (!ar.success) {
                oss << " err='" << ar.error_message << "'";
            }
            if (phase != nullptr) {
                oss << " phase='" << phase << "'";
            }
            traceLog(oss.str());
        }

        applyResidualAdditionAndConstraints();
        traceResidualComponents(phase);
        return computeResidualNorm();
    };

    auto assembleJacobianOnly = [&](const systems::SystemStateView& state) {
        auto J_view = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix) op='" + options_.jacobian_op + "'");
        }
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.is_nonlinear_iteration = true;
        const auto aj = transient.assemble(req, state, J_view.get(), nullptr);
        FE_THROW_IF(!aj.success, FEException,
                    "NewtonSolver: jacobian assembly failed: " + aj.error_message);
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=1 want_vector=0"
                << " ok=" << aj.success
                << " elems=" << aj.elements_assembled
                << " mat_ins=" << aj.matrix_entries_inserted
                << " time=" << aj.elapsed_time_seconds;
            if (!aj.success) {
                oss << " err='" << aj.error_message << "'";
            }
            traceLog(oss.str());
        }
    };

    auto assembleJacobianAndResidual = [&](const systems::SystemStateView& state) -> double {
        residual_op_used = options_.residual_op;
        auto J_view = J.createAssemblyView();
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix+vector) op='" + options_.residual_op + "'");
        }
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, J_view.get(), r_view.get());
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: combined (matrix+vector) assembly failed: " + ar.error_message);
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=1 want_vector=1"
                << " ok=" << ar.success
                << " elems=" << ar.elements_assembled
                << " mat_ins=" << ar.matrix_entries_inserted
                << " vec_ins=" << ar.vector_entries_inserted
                << " time=" << ar.elapsed_time_seconds;
            if (!ar.success) {
                oss << " err='" << ar.error_message << "'";
            }
            traceLog(oss.str());
        }

        applyResidualAdditionAndConstraints();
        traceResidualComponents("jacobian_and_residual");
        return computeResidualNorm();
    };

    auto assembleJacobianAndResidualWithJacobianOp = [&](const systems::SystemStateView& state,
                                                         bool& out_vector_ok) -> double {
        out_vector_ok = false;

        auto J_view = J.createAssemblyView();
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix+vector) op='" + options_.jacobian_op + "'");
        }
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, J_view.get(), r_view.get());
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: combined (matrix+vector) assembly failed: " + ar.error_message);
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=1 want_vector=1"
                << " ok=" << ar.success
                << " elems=" << ar.elements_assembled
                << " mat_ins=" << ar.matrix_entries_inserted
                << " vec_ins=" << ar.vector_entries_inserted
                << " time=" << ar.elapsed_time_seconds;
            if (!ar.success) {
                oss << " err='" << ar.error_message << "'";
            }
            traceLog(oss.str());
        }

        out_vector_ok = (ar.vector_entries_inserted > 0);
        if (!out_vector_ok) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        residual_op_used = options_.jacobian_op;
        applyResidualAdditionAndConstraints();
        return computeResidualNorm();
    };

    auto bridgeRankOneUpdates = [&]() -> bool {
        const auto updates = transient.system().lastRankOneUpdates();
        if (updates.empty()) {
            linear.setRankOneUpdates({});
            return false;
        }
        const auto& bordered = transient.system().borderedCoupling();
        const bool force_explicit_rank_one =
            std::getenv("SVMP_FORCE_EXPLICIT_RANK_ONE") != nullptr;
        const bool use_native_rank_one_updates =
            linear.supportsNativeRankOneUpdates() && !force_explicit_rank_one;
        const bool force_explicit_matrix_assembly =
            bordered.active && bordered.n_aux > 0 && !use_native_rank_one_updates;
        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: " + std::to_string(updates.size()) +
                     " rank-1 updates from assembly"
                     + (force_explicit_matrix_assembly ? " (explicit matrix path)" : ""));
            for (std::size_t i = 0; i < updates.size(); ++i) {
                double v_norm_sq = 0.0;
                for (const auto& [dof, val] : updates[i].v) {
                    (void)dof;
                    v_norm_sq += static_cast<double>(val) * static_cast<double>(val);
                }
                std::ostringstream oss;
                oss << "NewtonSolver: rank-1 update[" << i << "]"
                    << " sigma=" << updates[i].sigma
                    << " ||v||=" << std::sqrt(v_norm_sq)
                    << " nnz=" << updates[i].v.size();
                traceLog(oss.str());
            }
        }
        if (use_native_rank_one_updates) {
            linear.setRankOneUpdates(updates);
            return false;
        }

        linear.setRankOneUpdates({});
        {
            // Assemble the direct feedthrough contribution explicitly into the
            // bordered Jacobian so the monolithic Newton operator is backend
            // independent and the bordered K^{-1}B solves see the same matrix.
            auto J_view = J.createAssemblyView();
            FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: rank-1 fallback view");
            J_view->beginAssemblyPhase();
            std::vector<GlobalIndex> col_dofs;
            std::vector<Real> row_vals;
            std::array<GlobalIndex, 1> row_dof{};
            for (const auto& upd : updates) {
                col_dofs.resize(upd.v.size());
                row_vals.resize(upd.v.size());
                for (std::size_t j = 0; j < upd.v.size(); ++j) {
                    col_dofs[j] = upd.v[j].first;
                }
                for (const auto& ri : upd.v) {
                    row_dof[0] = ri.first;
                    const Real scale = upd.sigma * ri.second;
                    for (std::size_t j = 0; j < upd.v.size(); ++j) {
                        row_vals[j] = scale * upd.v[j].second;
                    }
                    J_view->addMatrixEntries(
                        std::span<const GlobalIndex>(row_dof.data(), row_dof.size()),
                        std::span<const GlobalIndex>(col_dofs.data(), col_dofs.size()),
                        std::span<const Real>(row_vals.data(), row_vals.size()),
                        assembly::AddMode::Add);
                }
            }
            J_view->finalizeAssembly();
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: rank-1 updates assembled directly into matrix");
            }
        }
        return true;
    };

    auto tolerancesSatisfied = [&](double norm) -> bool {
        const bool abs_enabled = options_.abs_tolerance > 0.0;
        const bool rel_enabled = options_.rel_tolerance > 0.0;
        const bool abs_ok = !abs_enabled || norm <= options_.abs_tolerance;
        const bool rel_ok = !rel_enabled
            || (report.residual_norm0 > 0.0
                    ? (norm / report.residual_norm0 <= options_.rel_tolerance)
                    : abs_ok);
        // All enabled nonlinear tolerances must be satisfied. This avoids
        // reporting convergence solely from relative reduction after a blow-up
        // or solely from an absolute check when a requested relative reduction
        // has not been achieved yet.
        return abs_ok && rel_ok;
    };

    auto assembleDtOnlyJacobianAndLumpedDiagonal = [&](const systems::SystemStateView& state) -> bool {
        if (!ptc_can_run) {
            return false;
        }

        auto* mass_lumped = workspace.ptc_mass_lumped.get();
        FE_CHECK_NOT_NULL(mass_lumped, "NewtonSolver: PTC mass lumped vector");

        const int max_order = transient.system().temporalOrder();
        if (max_order <= 0) {
            return false;
        }

        auto ctx_base = transient.integrator().buildContext(max_order, state);
        assembly::TimeIntegrationContext ctx_dt_only = ctx_base;
        ctx_dt_only.time_derivative_term_weight = static_cast<Real>(1.0);
        ctx_dt_only.non_time_derivative_term_weight = static_cast<Real>(0.0);

        systems::SystemStateView state_dt = state;
        state_dt.time_integration = &ctx_dt_only;

        J.zero();
        auto J_view = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: PTC dt-only Jacobian view");

        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.zero_outputs = true;
        req.suppress_constraint_inhomogeneity = true;
        const auto ar = transient.system().assemble(req, state_dt, J_view.get(), /*vector_out=*/nullptr);
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: PTC dt-only Jacobian assembly failed: " + ar.error_message);

        // Lump: m = A_dt * 1  (row sums of dt-only Jacobian).
        residual_scratch.set(static_cast<Real>(1.0));
        residual_scratch.updateGhosts();
        mass_lumped->zero();
        J.mult(residual_scratch, *mass_lumped);
        ptc_mass_ready = true;
        return true;
    };

    auto applyPtcDiagonalShift = [&](double target_gamma) {
        if (!ptc_can_run || !ptc_mass_ready) {
            return;
        }
        const double clamped = std::clamp(target_gamma, 0.0, options_.pseudo_transient.gamma_max);
        const double delta_gamma = clamped - ptc_gamma_applied;
        if (delta_gamma == 0.0) {
            ptc_gamma_applied = clamped;
            return;
        }

        auto* mass_lumped = workspace.ptc_mass_lumped.get();
        FE_CHECK_NOT_NULL(mass_lumped, "NewtonSolver: PTC mass lumped vector");
        auto m_view = mass_lumped->createAssemblyView();
        FE_CHECK_NOT_NULL(m_view.get(), "NewtonSolver: PTC mass view");

        auto J_mod = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_mod.get(), "NewtonSolver: PTC matrix modify view");
        J_mod->beginAssemblyPhase();
        for (const auto dof : ptc_owned_dofs) {
            const Real m = m_view->getVectorEntry(dof);
            const double md = std::abs(static_cast<double>(m));
            if (!(md > 0.0) || !std::isfinite(md)) {
                continue;
            }
            const double v = delta_gamma * md;
            if (v == 0.0 || !std::isfinite(v)) {
                continue;
            }
            J_mod->addMatrixEntry(dof, dof, static_cast<Real>(v), assembly::AddMode::Add);
        }
        J_mod->finalizeAssembly();
        ptc_gamma_applied = clamped;
    };

    bool have_residual = false;
    double current_residual_norm = std::numeric_limits<double>::quiet_NaN();
    bool have_jacobian = false;
    int last_jacobian_it = -1;
    const int jacobian_period = std::max(1, options_.jacobian_rebuild_period);

    // ===== NEWTON TIMING PROFILE =====
#ifdef SVMP_FE_ASSEMBLY_TIMING
    auto NTP = []() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
#else
    auto NTP = []() -> double { return 0.0; };
#endif
    double ntp_assembly = 0.0, ntp_linear = 0.0, ntp_update = 0.0;
    double ntp_constraints = 0.0, ntp_other = 0.0;
    double ntp_total_start = NTP();
    double ntp0;
    int ntp_assembly_count = 0, ntp_linear_iters_total = 0;
    auto printNewtonProfile = [&](int newton_iters) {
#ifdef SVMP_FE_ASSEMBLY_TIMING
        double ntp_total = NTP() - ntp_total_start;
        ntp_other = ntp_total - ntp_assembly - ntp_linear - ntp_update - ntp_constraints;
        if (ntp_other < 0.0) ntp_other = 0.0;
        int mpi_rank = 0;
#if FE_HAS_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
        if (mpi_rank == 0 && ntp_total > 1e-6) {
            auto pct = [&](double t) { return 100.0 * t / ntp_total; };
            fprintf(stderr,
              "\n+++ NEWTON SOLVER TIMING (rank 0) +++\n"
              "  Total Newton time:    %10.6f s  (%d Newton iters, %d assemblies, %d linear iters)\n"
              "  Assembly (J+r):       %10.6f s  (%5.1f%%)\n"
              "  Linear solve:         %10.6f s  (%5.1f%%)\n"
              "  Solution update:      %10.6f s  (%5.1f%%)\n"
              "  Constraint/ghosts:    %10.6f s  (%5.1f%%)\n"
              "  Other (overhead):     %10.6f s  (%5.1f%%)\n"
              "+++++++++++++++++++++++++++++++++++++++\n",
              ntp_total, newton_iters, ntp_assembly_count, ntp_linear_iters_total,
              ntp_assembly, pct(ntp_assembly),
              ntp_linear, pct(ntp_linear),
              ntp_update, pct(ntp_update),
              ntp_constraints, pct(ntp_constraints),
              ntp_other, pct(ntp_other));
        }
#else
        (void)newton_iters;
#endif
    };
    // =================================

    double prev_residual_norm = -1.0;

    for (int it = 0; it < max_it; ++it) {
        ntp0 = NTP();
        history.updateGhosts();

        if (!constraints.empty()) {
            constraints.distribute(history.u());
            history.u().updateGhosts();
        }
        ntp_constraints += NTP() - ntp0;

        auto state_holder = makeNewtonState(history, solve_time);
        const auto& state = state_holder.view;

        if (have_residual && !std::isfinite(current_residual_norm)) {
            // If the cached residual norm is invalid (e.g., NaN from a failed evaluation),
            // fall back to re-assembling the residual at the current state.
            have_residual = false;
        }

        const bool need_jacobian = !have_jacobian || (jacobian_period == 1) || ((it - last_jacobian_it) >= jacobian_period);
        bool jacobian_ready = have_jacobian && !need_jacobian;
        if (!have_residual) {
            ntp0 = NTP();
            if (need_jacobian && options_.assemble_both_when_possible && same_op) {
                // Residual and Jacobian share the same operator tag, so we can assemble both in one pass.
                current_residual_norm = assembleJacobianAndResidual(state);
                ptc_gamma_applied = 0.0;
                jacobian_ready = true;
                have_jacobian = true;
                last_jacobian_it = it;
            } else {
                // When residual_op != jacobian_op, always assemble the residual using residual_op so
                // Newton convergence checks and line search evaluate the same residual used in the
                // linear solve. (Some modules may also install vector contributions under jacobian_op
                // as an optimization; those must not silently change the residual definition.)
                current_residual_norm = assembleResidualOnly(state, /*phase=*/nullptr);
                if (need_jacobian) {
                    assembleJacobianOnly(state);
                    ptc_gamma_applied = 0.0;
                    jacobian_ready = true;
                    have_jacobian = true;
                    last_jacobian_it = it;
                }
            }
            ntp_assembly += NTP() - ntp0;
            ntp_assembly_count++;
            have_residual = true;
        } else if (need_jacobian && options_.assemble_both_when_possible && same_op &&
                   has_monolithic_auxiliary_unknowns) {
            ntp0 = NTP();
            current_residual_norm = assembleJacobianAndResidual(state);
            ptc_gamma_applied = 0.0;
            jacobian_ready = true;
            have_jacobian = true;
            last_jacobian_it = it;
            have_residual = true;
            ntp_assembly += NTP() - ntp0;
            ntp_assembly_count++;
        }

        report.residual_norm = current_residual_norm;
        if (it == 0) {
            report.residual_norm0 = current_residual_norm;
        }

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            const double denom = (report.residual_norm0 > 0.0) ? report.residual_norm0 : 1.0;
            oss << "NewtonSolver: it=" << it
                << " ||r||=" << report.residual_norm
                << " ||r0||=" << report.residual_norm0
                << " rel=" << (report.residual_norm / denom);
            traceLog(oss.str());
        }

        // Nullspace validation: on the first iteration with a Jacobian, optionally
        // verify that inferred nullspace vectors are actually in the operator's nullspace.
        // Gated by SVMP_GAUGE_VALIDATE environment variable to avoid overhead in production.
        if (it == 0 && have_jacobian && gauge::isNullspaceValidationEnabled()) {
            const auto* reg = transient.system().gaugeRegistryIfPresent();
            if (reg && reg->isResolved()) {
                const auto n_dofs = transient.system().dofHandler().getNumDofs();
                auto get_field_dofs = [&](FieldId fid, int /*comp*/) -> std::vector<GlobalIndex> {
                    const auto offset = transient.system().fieldDofOffset(fid);
                    const auto& fdh = transient.system().fieldDofHandler(fid);
                    const auto nd = fdh.getNumDofs();
                    std::vector<GlobalIndex> dofs;
                    dofs.reserve(static_cast<std::size_t>(nd));
                    for (GlobalIndex d = offset; d < offset + nd; ++d) dofs.push_back(d);
                    return dofs;
                };
                // Build basis from ALL resolved modes (not just SolverNullspace)
                // by temporarily treating all ExactNullspace modes as needing basis
                auto all_basis = reg->buildNullspaceBasis(n_dofs, get_field_dofs);
                if (!all_basis.empty()) {
                    auto validation_factory = backends::BackendFactory::create(J.backendKind());
                    if (validation_factory) {
                        auto results = gauge::validateNullspaceBasis(
                            J, *validation_factory, all_basis);
                        std::fprintf(stderr, "%s",
                            gauge::formatValidationReport(results).c_str());
                    }
                }
            }
        }

        if (tolerancesSatisfied(current_residual_norm)) {
            report.converged = true;
            report.iterations = it;
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: converged before linear solve (tolerances satisfied).");
            }
            printNewtonProfile(it);
            return report;
        }

        // Stagnation detection is diagnostic-only unless the configured
        // nonlinear tolerances are also satisfied. This prevents the solver
        // from reporting "converged" at residual levels that are still far
        // above the requested abs/rel tolerances.
        if (it > 0 && options_.stagnation_tolerance > 0.0 &&
            prev_residual_norm > 0.0 && std::isfinite(prev_residual_norm) &&
            report.residual_norm0 > 0.0 && current_residual_norm < report.residual_norm0) {
            const double ratio = current_residual_norm / prev_residual_norm;
            if (ratio >= options_.stagnation_tolerance) {
                if (tolerancesSatisfied(current_residual_norm)) {
                    report.converged = true;
                    report.iterations = it;
                    if (oopTraceEnabled()) {
                        std::ostringstream oss;
                        oss << "NewtonSolver: converged by stagnation within tolerance"
                            << " (||r_k||/||r_{k-1}||=" << ratio
                            << " >= " << options_.stagnation_tolerance << ")";
                        traceLog(oss.str());
                    }
                    printNewtonProfile(it);
                    return report;
                }
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: stagnation detected but residual remains above tolerance"
                        << " (||r_k||/||r_{k-1}||=" << ratio
                        << " >= " << options_.stagnation_tolerance
                        << ", ||r||=" << current_residual_norm << ")";
                    traceLog(oss.str());
                }
            }
        }
        prev_residual_norm = current_residual_norm;

        if (ptc_can_run) {
            if (options_.pseudo_transient.update_from_residual_ratio && ptc_mass_ready &&
                std::isfinite(ptc_prev_residual_norm) && ptc_prev_residual_norm > 0.0 &&
                std::isfinite(current_residual_norm) && current_residual_norm >= 0.0) {
                const double ratio = current_residual_norm / ptc_prev_residual_norm;
                if (std::isfinite(ratio) && ratio > 0.0) {
                    ptc_gamma = std::min(ptc_gamma * ratio, options_.pseudo_transient.gamma_max);
                    if (ptc_gamma < options_.pseudo_transient.gamma_drop_tolerance) {
                        ptc_gamma = 0.0;
                    }
                }
            }
            ptc_prev_residual_norm = current_residual_norm;
        }

        if (need_jacobian && !jacobian_ready) {
            assembleJacobianOnly(state);
            ptc_gamma_applied = 0.0;
            have_jacobian = true;
            last_jacobian_it = it;
        }

        if (jacobianCheckEnabled() && need_jacobian && it == jacobianCheckNewtonIteration()) {
            // Directional finite-difference check: compare J*v (from `jacobian_op`) to
            // (r(u+h*v)-r(u))/h (assembled with `residual_op`).
            //
            // This is a lightweight runtime diagnostic for missing/incomplete Jacobians *and*
            // operator mismatches between the configured residual and Jacobian operators.
            const double rel_step = jacobianCheckRelativeStep();
            const int n_dofs = sys.dofHandler().getNumDofs();
            const double u_norm = history.u().norm();
            const double u_rms = (n_dofs > 0) ? (u_norm / std::sqrt(static_cast<double>(n_dofs))) : u_norm;
            const double h = rel_step * (1.0 + u_rms);

            if (h > 0.0 && std::isfinite(h)) {
                // Populate a deterministic pseudo-random direction in `du` (will be overwritten by the linear solve).
                {
                    auto v = du.localSpan();
                    std::uint64_t s = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(mpiRank() + 1);
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        // xorshift64*
                        s ^= s >> 12;
                        s ^= s << 25;
                        s ^= s >> 27;
                        const std::uint64_t x = s * 2685821657736338717ULL;
                        const double u01 = static_cast<double>((x >> 11) & ((1ULL << 53) - 1ULL)) *
                            (1.0 / 9007199254740992.0); // 2^53
                        v[i] = static_cast<Real>(2.0 * u01 - 1.0);
                    }
                }
                zeroVectorEntries(constrained_dofs, du);
                du.updateGhosts();

                auto applyResidualFixups = [&](backends::GenericVector& vec) {
                    if (residual_addition != nullptr) {
                        axpy(vec, static_cast<Real>(1.0), *residual_addition);
                    }
                    zeroVectorEntries(constrained_dofs, vec);
                };

                // Backup the current nonlinear state so the diagnostic
                // assemblies do not perturb the live monolithic bordered data.
                copyVector(u_backup, history.u());
                const auto aux_state_backup =
                    transient.system().checkpointAuxiliaryState();
                const auto bordered_backup = transient.system().borderedCoupling();

                auto restoreDiagnosticState = [&]() {
                    copyVector(history.u(), u_backup);
                    if (!constraints.empty()) {
                        constraints.distribute(history.u());
                    }
                    history.u().updateGhosts();
                    if (!aux_state_backup.empty()) {
                        transient.system().restoreAuxiliaryState(aux_state_backup);
                    }
                    transient.system().borderedCoupling() = bordered_backup;
                    if (auto* reg = transient.system().auxiliaryInputRegistryIfPresent()) {
                        reg->invalidateAll();
                    }
                };

                // Assemble r(u) with residual_op into residual_base.
                residual_base.zero();
                {
                    auto r_view = residual_base.createAssemblyView();
                    FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: jacobian check residual base view");

                    transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                                     /*invalidate_auxiliary_inputs=*/false);
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    req.suppress_constraint_inhomogeneity = true;
                    req.is_nonlinear_iteration = true;
                    const auto ar = transient.assemble(req, state, nullptr, r_view.get());
                    FE_THROW_IF(!ar.success, FEException,
                                "NewtonSolver: Jacobian check base residual assembly failed: " + ar.error_message);
                }
                applyResidualFixups(residual_base);
                restoreDiagnosticState();

                // Assemble r(u + h*v) with residual_op into residual_scratch.
                axpy(history.u(), static_cast<Real>(h), du);
                if (!constraints.empty()) {
                    constraints.distribute(history.u());
                }
                history.u().updateGhosts();

                residual_scratch.zero();
                {
                    auto r_view = residual_scratch.createAssemblyView();
                    FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: jacobian check residual perturbed view");

                    transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                                     /*invalidate_auxiliary_inputs=*/false);
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    req.suppress_constraint_inhomogeneity = true;
                    req.is_nonlinear_iteration = true;
                    auto perturbed_state_holder = makeNewtonState(history, solve_time);
                    const auto ar = transient.assemble(
                        req, perturbed_state_holder.view, nullptr, r_view.get());
                    FE_THROW_IF(!ar.success, FEException,
                                "NewtonSolver: Jacobian check perturbed residual assembly failed: " + ar.error_message);
                }
                applyResidualFixups(residual_scratch);
                restoreDiagnosticState();

                const double r_base_norm = residualNormForConvergence(residual_base, u_backup);
                const double r_used_norm = residualNormForConvergence(r, u_backup);

                // residual_scratch <- (r(u+h*v) - r(u)) / h  (FD approximation of J*v).
                axpy(residual_scratch, static_cast<Real>(-1.0), residual_base);
                residual_scratch.scale(static_cast<Real>(1.0 / h));

                // u_backup <- r_used - r_base (will overwrite u_backup).
                copyVector(u_backup, r);
                axpy(u_backup, static_cast<Real>(-1.0), residual_base);
                zeroVectorEntries(constrained_dofs, u_backup);
                const double r_diff_norm = residualNormForConvergence(u_backup, residual_base);

                // u_backup <- J*v (FSILS matvec applies overlap communication).
                u_backup.zero();
                J.mult(du, u_backup);
                zeroVectorEntries(constrained_dofs, u_backup);
                const double jv_norm = u_backup.norm();

                // The FD residual is assembled by element ownership; sum overlap contributions once for comparison.
                accumulateOverlapIfNeeded(residual_scratch);
                const double fd_norm = residual_scratch.norm();

                // u_backup <- J*v - FD
                axpy(u_backup, static_cast<Real>(-1.0), residual_scratch);
                const double err_norm = u_backup.norm();
                const double denom = std::max({jv_norm, fd_norm, 1e-14});
                const double rel_err = err_norm / denom;

                if (mpiRank() == 0) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: Jacobian check jacobian_op='" << options_.jacobian_op
                        << "' residual_op='" << options_.residual_op << "'"
                        << " it=" << it
                        << " h=" << h
                        << " ||Jv||=" << jv_norm
                        << " ||FD||=" << fd_norm
                        << " ||Jv-FD||=" << err_norm
                        << " rel=" << rel_err
                        << " ||r(residual_op)||=" << r_base_norm
                        << " ||r(used_op=" << residual_op_used << ")||=" << r_used_norm
                        << " ||r_used-r_residual||=" << r_diff_norm;
                    FE_LOG_INFO(oss.str());
                }
            } else if (mpiRank() == 0) {
                FE_LOG_INFO("NewtonSolver: Jacobian check skipped (invalid perturbation size).");
            }
        }

        du.zero();

        const bool ptc_always_on = ptc_can_run && !options_.pseudo_transient.activate_on_linear_failure &&
                                  (options_.pseudo_transient.gamma_initial > 0.0);
        if (ptc_always_on && !ptc_mass_ready) {
            // Assemble dt-only Jacobian to build a mass-like diagonal, then restore the physical Jacobian.
            (void)assembleDtOnlyJacobianAndLumpedDiagonal(state);

            if (options_.assemble_both_when_possible && same_op) {
                current_residual_norm = assembleJacobianAndResidual(state);
                have_residual = true;
                have_jacobian = true;
                last_jacobian_it = it;
            } else {
                current_residual_norm = assembleResidualOnly(state, /*phase=*/"ptc_restore");
                have_residual = true;
                assembleJacobianOnly(state);
                ptc_gamma_applied = 0.0;
                have_jacobian = true;
                last_jacobian_it = it;
            }
            ptc_gamma_applied = 0.0;
            ptc_gamma = options_.pseudo_transient.gamma_initial;
        }

        // Apply current PTC diagonal shift (may be zero).
        if (ptc_can_run && ptc_mass_ready) {
            applyPtcDiagonalShift(ptc_gamma);
        }

        // Bridge rank-1 updates from coupled BC assembly to the linear solver.
        const bool explicit_rank_one_in_matrix = bridgeRankOneUpdates();
        if (!constrained_dofs.empty()) {
            const auto& bordered = transient.system().borderedCoupling();
            const bool exact_direct_coupling_in_matrix =
                transient.system().lastRankOneUpdates().empty() &&
                !bordered.direct_coupling_records.empty();
            if (explicit_rank_one_in_matrix || exact_direct_coupling_in_matrix) {
                reapplyConstrainedJacobianRows();
            }
        }

        // Bridge nullspace basis from GaugeRegistry to the linear solver.
        // Currently dormant: the resolver always uses algebraic enforcement,
        // so buildNullspaceBasis() returns empty.  This path is retained for
        // future SolverNullspace opt-in.
        if (linear.supportsNullspace()) {
            const auto* reg = transient.system().gaugeRegistryIfPresent();
            if (reg && reg->isResolved()) {
                const auto n_dofs = transient.system().dofHandler().getNumDofs();
                auto get_field_dofs = [&](FieldId fid, int comp) -> std::vector<GlobalIndex> {
                    const auto idx = static_cast<std::size_t>(fid);
                    const auto& sys = transient.system();

                    // Component-aware: return only DOFs for the requested component
                    if (comp >= 0 && idx < sys.fieldMap().numFields()) {
                        const auto n_comp = sys.fieldMap().numComponents(idx);
                        if (n_comp > 1 && static_cast<LocalIndex>(comp) < n_comp) {
                            return sys.fieldMap().getComponentDofs(idx, static_cast<LocalIndex>(comp)).toVector();
                        }
                    }

                    const auto offset = sys.fieldDofOffset(fid);
                    const auto& fdh = sys.fieldDofHandler(fid);
                    const auto nd = fdh.getNumDofs();
                    std::vector<GlobalIndex> dofs;
                    dofs.reserve(static_cast<std::size_t>(nd));
                    for (GlobalIndex d = offset; d < offset + nd; ++d) {
                        dofs.push_back(d);
                    }
                    return dofs;
                };
                // Build CoordinateProvider for rotation mode basis vectors.
                gauge::GaugeRegistry::CoordinateProvider coord_provider;
                const auto* emap = transient.system().dofHandler().getEntityDofMap();
                if (emap) {
                    coord_provider = [&](FieldId /*fid*/, GlobalIndex dof)
                        -> std::array<double, 3> {
                        auto ent = emap->getDofEntity(dof);
                        if (ent && ent->kind == dofs::EntityKind::Vertex) {
                            auto p = transient.system().meshAccess().getNodeCoordinates(ent->id);
                            return {static_cast<double>(p[0]),
                                    static_cast<double>(p[1]),
                                    static_cast<double>(p[2])};
                        }
                        return {0.0, 0.0, 0.0};
                    };
                }
                auto basis = reg->buildNullspaceBasis(n_dofs, get_field_dofs, coord_provider);
                linear.setNullspaceBasis(basis);
            }
        }

        // Provide the effective stage time step to the linear solver backend.
        //
        // For multi-stage schemes (e.g., generalized-α), `solve_time` may be a stage time
        // t_{n+α}. The legacy solver scales certain coupled-BC linearization terms by the
        // stage step (α*dt). Passing this here allows backends like FSILS to apply the same
        // scaling internally without coupling the FE library to specific physics.
        double dt_eff = base_state.dt;
        const double stage_dt = solve_time - history.time();
        if (std::isfinite(stage_dt) && stage_dt > 0.0) {
            dt_eff = stage_dt;
        }
        linear.setEffectiveTimeStep(dt_eff);
        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: effective dt for linear backend=" + std::to_string(dt_eff));
        }

        const auto& bordered = transient.system().borderedCoupling();
        const bool has_bordered = bordered.active && bordered.n_aux > 0;
        std::vector<Real> aux_delta;
        FsilsMatrixSnapshot fsils_matrix_snapshot;

        int ptc_retries = 0;
        while (true) {
            SolverOptionsGuard bordered_solver_options_guard{linear, linear.getOptions()};
            if (has_bordered) {
                linear.setOptions(makeBorderedSolveOptions(bordered_solver_options_guard.saved));
                fsils_matrix_snapshot = captureFsilsMatrixSnapshot(J);
            }
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: calling linear.solve()");
            }
            ntp0 = NTP();
            report.linear = linear.solve(J, du, r);
            ntp_linear += NTP() - ntp0;
            ntp_linear_iters_total += report.linear.iterations;
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: linear solve converged=" << report.linear.converged
                    << " iters=" << report.linear.iterations
                    << " r0=" << report.linear.initial_residual_norm
                    << " rn=" << report.linear.final_residual_norm
                    << " rel=" << report.linear.relative_residual
                    << " msg='" << report.linear.message << "'";
                traceLog(oss.str());
            }
            if (report.linear.converged) {
                break;
            }

            // Inexact Newton: accept the approximate solution even when the linear
            // solve doesn't fully converge. This matches the legacy solver behavior
            // where imprecise linear solutions still produce effective Newton steps.
            const bool allow_inexact_main_solve =
                options_.accept_inexact_linear_solutions && !has_bordered;
            if (allow_inexact_main_solve) {
                if (oopTraceEnabled()) {
                    traceLog("NewtonSolver: accepting inexact linear solution (rel=" +
                             std::to_string(report.linear.relative_residual) + ")");
                }
                break;
            }

            const bool can_activate_ptc = ptc_can_run && options_.pseudo_transient.activate_on_linear_failure;
            if (!can_activate_ptc) {
                FE_THROW(FEException, "NewtonSolver: linear solve did not converge: " + report.linear.message);
            }

            // Lazily build the dt-only lumped diagonal when first needed.
            if (!ptc_mass_ready) {
                (void)assembleDtOnlyJacobianAndLumpedDiagonal(state);

                // Restore the physical Jacobian (dt-only assembly overwrote `J`).
                if (options_.assemble_both_when_possible && same_op) {
                    current_residual_norm = assembleJacobianAndResidual(state);
                    have_residual = true;
                    have_jacobian = true;
                    last_jacobian_it = it;
                } else {
                    current_residual_norm = assembleResidualOnly(state, /*phase=*/"ptc_restore");
                    have_residual = true;
                    assembleJacobianOnly(state);
                    ptc_gamma_applied = 0.0;
                    have_jacobian = true;
                    last_jacobian_it = it;
                }
                ptc_gamma_applied = 0.0;
            }

            // Increase diagonal dominance and retry.
            if (!(ptc_gamma > 0.0)) {
                ptc_gamma = (options_.pseudo_transient.gamma_initial > 0.0)
                                ? options_.pseudo_transient.gamma_initial
                                : 1.0;
            } else {
                ptc_gamma = std::min(ptc_gamma * options_.pseudo_transient.gamma_growth,
                                     options_.pseudo_transient.gamma_max);
            }

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: PTC retry linear solve (gamma=" << ptc_gamma
                    << ", retry=" << (ptc_retries + 1) << "/" << options_.pseudo_transient.max_linear_retries << ")";
                traceLog(oss.str());
            }

            applyPtcDiagonalShift(ptc_gamma);

            ++ptc_retries;
            FE_THROW_IF(ptc_retries >= options_.pseudo_transient.max_linear_retries, FEException,
                        "NewtonSolver: linear solve did not converge (PTC retries exhausted): " + report.linear.message);
            du.zero();
        }

        if (has_bordered) {
            const auto nf = bordered.n_field_dofs;
            const auto na = static_cast<std::size_t>(bordered.n_aux);
            FE_THROW_IF(nf != static_cast<std::size_t>(du.size()), systems::InvalidStateException,
                        "NewtonSolver: bordered PDE block size does not match solution size");
            FE_THROW_IF(bordered.B.size() != nf * na ||
                            bordered.Ct.size() != nf * na ||
                            bordered.D.size() != na * na ||
                            bordered.g.size() != na,
                        systems::InvalidStateException,
                        "NewtonSolver: bordered coupling storage size mismatch");

            auto gatherDenseVector = [](backends::GenericVector& vec, std::size_t n) {
                auto view = vec.createAssemblyView();
                FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: bordered dense gather view");
                std::vector<Real> dense(n, 0.0);
                for (std::size_t k = 0; k < n; ++k) {
                    dense[k] = view->getVectorEntry(static_cast<GlobalIndex>(k));
                }
                return dense;
            };

            auto scatterDenseVector = [](backends::GenericVector& vec, std::span<const Real> dense) {
                auto view = vec.createAssemblyView();
                FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: bordered dense scatter view");
                view->beginAssemblyPhase();
                for (std::size_t k = 0; k < dense.size(); ++k) {
                    view->addVectorEntry(static_cast<GlobalIndex>(k),
                                         dense[k],
                                         assembly::AddMode::Insert);
                }
                view->finalizeAssembly();
            };

            copyVector(residual_base, du);
            const auto u0 = gatherDenseVector(residual_base, nf);
            std::vector<Real> z_columns(nf * na, 0.0);

            {
                SolverOptionsGuard bordered_solver_options_guard{linear, linear.getOptions()};
                linear.setOptions(makeBorderedSolveOptions(bordered_solver_options_guard.saved));

                for (std::size_t j = 0; j < na; ++j) {
                    restoreFsilsMatrixSnapshot(J, fsils_matrix_snapshot);

                    residual_scratch.zero();
                    {
                        auto rhs_view = residual_scratch.createAssemblyView();
                        FE_CHECK_NOT_NULL(rhs_view.get(), "NewtonSolver: bordered rhs view");
                        rhs_view->beginAssemblyPhase();
                        for (std::size_t row = 0; row < nf; ++row) {
                            const Real bij = bordered.B[row + nf * j];
                            if (std::abs(bij) > Real(1e-30)) {
                                rhs_view->addVectorEntry(static_cast<GlobalIndex>(row),
                                                         bij,
                                                         assembly::AddMode::Add);
                            }
                        }
                        rhs_view->finalizeAssembly();
                    }

                    du.zero();
                    ntp0 = NTP();
                    const auto z_report = linear.solve(J, du, residual_scratch);
                    ntp_linear += NTP() - ntp0;
                    ntp_linear_iters_total += z_report.iterations;
                    FE_THROW_IF(!z_report.converged, FEException,
                                "NewtonSolver: bordered K^{-1}B solve did not converge: " +
                                    z_report.message);

                    const auto z_col = gatherDenseVector(du, nf);
                    for (std::size_t row = 0; row < nf; ++row) {
                        z_columns[j * nf + row] = z_col[row];
                    }

                    if (oopTraceEnabled()) {
                        const auto z_norm = std::sqrt(std::inner_product(
                            z_col.begin(), z_col.end(), z_col.begin(), Real(0.0)));
                        std::ostringstream oss;
                        oss << "NewtonSolver: bordered column " << j
                            << " ||K^{-1}B_j||=" << z_norm
                            << " iters=" << z_report.iterations;
                        traceLog(oss.str());
                    }
                }
            }

            std::vector<Real> schur = bordered.D;
            for (std::size_t i = 0; i < na; ++i) {
                for (std::size_t j = 0; j < na; ++j) {
                    Real ctz = 0.0;
                    for (std::size_t k = 0; k < nf; ++k) {
                        ctz += bordered.Ct[i * nf + k] * z_columns[j * nf + k];
                    }
                    schur[i * na + j] -= ctz;
                }
            }

            aux_delta = bordered.g;
            for (std::size_t i = 0; i < na; ++i) {
                for (std::size_t k = 0; k < nf; ++k) {
                    aux_delta[i] -= bordered.Ct[i * nf + k] * u0[k];
                }
            }

            FE_THROW_IF(!solveDenseLinearSystem(schur, aux_delta), systems::InvalidStateException,
                        "NewtonSolver: bordered Schur solve failed");

            std::vector<Real> dense_du = u0;
            for (std::size_t j = 0; j < na; ++j) {
                const Real dxj = aux_delta[j];
                for (std::size_t k = 0; k < nf; ++k) {
                    dense_du[k] -= z_columns[j * nf + k] * dxj;
                }
            }
            scatterDenseVector(du, dense_du);

            if (oopTraceEnabled()) {
                const auto dx_norm = std::sqrt(std::inner_product(
                    aux_delta.begin(), aux_delta.end(), aux_delta.begin(), Real(0.0)));
                std::ostringstream oss;
                oss << "NewtonSolver: bordered correction ||dx_aux||=" << dx_norm;
                traceLog(oss.str());

                J.mult(du, residual_base);
                const auto Kdu = gatherDenseVector(residual_base, nf);
                const auto rhs_dense = gatherDenseVector(r, nf);

                double pde_lin_res_sq = 0.0;
                for (std::size_t i = 0; i < nf; ++i) {
                    Real val = Kdu[i];
                    for (std::size_t j = 0; j < na; ++j) {
                        val += bordered.B[i + nf * j] * aux_delta[j];
                    }
                    const double rem = static_cast<double>(val - rhs_dense[i]);
                    pde_lin_res_sq += rem * rem;
                }

                double aux_lin_res_sq = 0.0;
                for (std::size_t i = 0; i < na; ++i) {
                    Real val = Real(0.0);
                    for (std::size_t k = 0; k < nf; ++k) {
                        val += bordered.Ct[i * nf + k] * dense_du[k];
                    }
                    for (std::size_t j = 0; j < na; ++j) {
                        val += bordered.D[i * na + j] * aux_delta[j];
                    }
                    const double rem = static_cast<double>(val - bordered.g[i]);
                    aux_lin_res_sq += rem * rem;
                }

                std::ostringstream lin_oss;
                lin_oss << "NewtonSolver: bordered linear residual"
                        << " pde=" << std::sqrt(pde_lin_res_sq)
                        << " aux=" << std::sqrt(aux_lin_res_sq)
                        << " mixed=" << std::sqrt(pde_lin_res_sq + aux_lin_res_sq);
                traceLog(lin_oss.str());
            }
        }

        auto applyDtIncrementScaling = [&]() {
            if (!options_.scale_dt_increments || workspace.dt_field_dofs.empty()) {
                return;
            }
            double factor = options_.dt_increment_scale;
            if (!(factor > 0.0)) {
                const auto* time_ctx = dt_scale_ctx ? &(*dt_scale_ctx) : nullptr;
                if (time_ctx && time_ctx->dt1) {
                    const double a0 = static_cast<double>(time_ctx->dt1->coeff(/*history_index=*/0));
                    if (std::isfinite(a0) && std::abs(a0) > 0.0) {
                        factor = 1.0 / a0;
                    }
                }
            }
            if (factor > 0.0 && std::isfinite(factor) && std::abs(factor - 1.0) > 0.0) {
                auto du_view = du.createAssemblyView();
                FE_CHECK_NOT_NULL(du_view.get(), "NewtonSolver: du scaling view");
                du_view->beginAssemblyPhase();
                for (const auto dof : workspace.dt_field_dofs) {
                    const Real v = du_view->getVectorEntry(dof);
                    du_view->addVectorEntry(dof, static_cast<Real>(factor) * v, assembly::AddMode::Insert);
                }
                du_view->finalizeAssembly();
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: scaled dt increments by factor=" << factor
                        << " dofs=" << workspace.dt_field_dofs.size();
                    traceLog(oss.str());
                }
            }
        };

        applyDtIncrementScaling();

        const double du_norm = du.norm();

        if (!options_.use_line_search) {
            ntp0 = NTP();
            if (!aux_delta.empty()) {
                applyAuxiliaryDelta(transient.system(), bordered, aux_delta, static_cast<Real>(1.0));
            }
            axpy(history.u(), static_cast<Real>(-1.0), du);
            if (!constraints.empty()) {
                constraints.distribute(history.u());
            }
            history.u().updateGhosts();
            ntp_update += NTP() - ntp0;
            have_residual = false;

            if (options_.step_tolerance > 0.0) {
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: step ||du||=" << du_norm << " step_tol=" << options_.step_tolerance;
                    traceLog(oss.str());
                }
                if (du_norm <= options_.step_tolerance) {
                    report.converged = true;
                    report.iterations = it + 1;
                    if (oopTraceEnabled()) {
                        traceLog("NewtonSolver: converged by step tolerance.");
                    }
                    printNewtonProfile(it + 1);
                    return report;
                }
            }
            continue;
        }

        // Backtracking line search: choose alpha in (0,1] so the residual norm decreases.
        copyVector(u_backup, history.u());
        const auto aux_state_backup =
            aux_delta.empty() ? std::vector<Real>{} : transient.system().checkpointAuxiliaryState();
        const double r_norm0 = current_residual_norm;
        const double r_norm0_sq = r_norm0 * r_norm0;

        double alpha = 1.0;
        double trial_norm = std::numeric_limits<double>::infinity();
        bool accepted = false;
        double best_alpha = 0.0;
        double best_trial_norm = std::numeric_limits<double>::infinity();
        bool have_best_trial = false;

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: line search begin alpha=1"
                << " alpha_min=" << options_.line_search_alpha_min
                << " shrink=" << options_.line_search_shrink
                << " c1=" << options_.line_search_c1;
            traceLog(oss.str());
        }

        for (int ls = 0; ls < options_.line_search_max_iterations; ++ls) {
            copyVector(history.u(), u_backup);
            if (!aux_state_backup.empty()) {
                transient.system().restoreAuxiliaryState(aux_state_backup);
                applyAuxiliaryDelta(transient.system(), bordered, aux_delta, static_cast<Real>(alpha));
            }
            axpy(history.u(), static_cast<Real>(-alpha), du);
            if (!constraints.empty()) {
                constraints.distribute(history.u());
            }
            history.u().updateGhosts();

            auto trial_state_holder = makeNewtonState(history, solve_time);
            trial_norm = assembleResidualOnly(trial_state_holder.view, /*phase=*/"line_search");

            bool ok = false;
            if (std::isfinite(trial_norm) && std::isfinite(r_norm0)) {
                if (has_bordered) {
                    // For monolithically coupled auxiliary states, use the full
                    // nonlinear residual as the merit function and accept any
                    // trial that decreases it. This avoids PDE-centric Armijo
                    // rejection of otherwise good bordered Newton steps.
                    ok = (trial_norm <= r_norm0 * (1.0 + 1e-12));
                } else {
                    // Armijo on phi(u) = 0.5*||r(u)||^2 with Newton direction.
                    const double rhs = (1.0 - 2.0 * options_.line_search_c1 * alpha) * r_norm0_sq;
                    if (rhs > 0.0) {
                        ok = (trial_norm * trial_norm <= rhs);
                    } else {
                        ok = (trial_norm <= r_norm0);
                    }
                }
            }

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: line search trial alpha=" << alpha
                    << " ||r(alpha)||=" << trial_norm
                    << " ok=" << (ok ? 1 : 0);
                traceLog(oss.str());
            }

            if (std::isfinite(trial_norm) && trial_norm < best_trial_norm) {
                best_trial_norm = trial_norm;
                best_alpha = alpha;
                have_best_trial = true;
            }

            if (ok) {
                accepted = true;
                break;
            }

            if (alpha <= options_.line_search_alpha_min) {
                break;
            }
            alpha *= options_.line_search_shrink;
            if (alpha < options_.line_search_alpha_min) {
                alpha = options_.line_search_alpha_min;
            }
        }

        if (!accepted) {
            if (have_best_trial && best_trial_norm <= r_norm0) {
                alpha = best_alpha;
                copyVector(history.u(), u_backup);
                if (!aux_state_backup.empty()) {
                    transient.system().restoreAuxiliaryState(aux_state_backup);
                    applyAuxiliaryDelta(transient.system(), bordered, aux_delta, static_cast<Real>(alpha));
                }
                axpy(history.u(), static_cast<Real>(-alpha), du);
                if (!constraints.empty()) {
                    constraints.distribute(history.u());
                }
                history.u().updateGhosts();
                auto best_state_holder = makeNewtonState(history, solve_time);
                trial_norm = assembleResidualOnly(
                    best_state_holder.view, /*phase=*/"line_search_best_fallback");
            } else {
                alpha = 0.0;
                copyVector(history.u(), u_backup);
                if (!aux_state_backup.empty()) {
                    transient.system().restoreAuxiliaryState(aux_state_backup);
                }
                if (auto* reg = transient.system().auxiliaryInputRegistryIfPresent()) {
                    reg->invalidateAll();
                }
                if (!constraints.empty()) {
                    constraints.distribute(history.u());
                }
                history.u().updateGhosts();
                auto restored_state_holder = makeNewtonState(history, solve_time);
                trial_norm = assembleResidualOnly(
                    restored_state_holder.view, /*phase=*/"line_search_restore");
            }
        }

        if (!accepted && oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: line search did not satisfy decrease; fallback alpha=" << alpha
                << " ||r(alpha)||=" << trial_norm;
            traceLog(oss.str());
        }

        // `history.u` and `r` already correspond to the last trial (accepted or fallback).
        current_residual_norm = trial_norm;
        have_residual = std::isfinite(current_residual_norm);

        if (options_.step_tolerance > 0.0) {
            const double step_norm = alpha * du_norm;
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: step ||alpha*du||=" << step_norm << " step_tol=" << options_.step_tolerance;
                traceLog(oss.str());
            }
            if (step_norm <= options_.step_tolerance) {
                report.converged = true;
                report.iterations = it + 1;
                report.residual_norm = current_residual_norm;
                if (oopTraceEnabled()) {
                    traceLog("NewtonSolver: converged by step tolerance.");
                }
                printNewtonProfile(it + 1);
                return report;
            }
        }

        if (tolerancesSatisfied(current_residual_norm)) {
            report.converged = true;
            report.iterations = it + 1;
            report.residual_norm = current_residual_norm;
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: converged after line search update (tolerances satisfied).");
            }
            printNewtonProfile(it + 1);
            return report;
        }
    }

    // When line search is disabled, we do not evaluate the residual norm after applying the
    // last Newton update (we normally do it at the start of the next iteration). If we
    // exit the loop due to reaching `max_it`, perform one final residual-only evaluation
    // so we don't incorrectly report "not converged" when the last update actually met
    // the requested tolerances.
    if (!have_residual) {
        history.updateGhosts();
        if (!constraints.empty()) {
            constraints.distribute(history.u());
            history.u().updateGhosts();
        }
        auto final_state_holder = makeNewtonState(history, solve_time);
        current_residual_norm = assembleResidualOnly(
            final_state_holder.view, /*phase=*/"final_check");
        have_residual = true;
        report.residual_norm = current_residual_norm;

        if (tolerancesSatisfied(current_residual_norm)) {
            report.converged = true;
            report.iterations = max_it;
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: converged after final update (tolerances satisfied).");
            }
            printNewtonProfile(max_it);
            return report;
        }
    }

    report.converged = false;
    report.iterations = max_it;
    if (have_residual && std::isfinite(current_residual_norm)) {
        report.residual_norm = current_residual_norm;
    }
    if (oopTraceEnabled()) {
        traceLog("NewtonSolver: reached max iterations without convergence.");
    }
    printNewtonProfile(max_it);
    return report;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
