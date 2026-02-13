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
#include "Core/Logger.h"
#include "Dofs/DofIndexSet.h"
#include "Systems/SystemsExceptions.h"

#if defined(FE_HAS_FSILS)
#  include "Backends/FSILS/FsilsVector.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <limits>
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

    history.updateGhosts();

    const systems::SystemStateView base_state = makeStateView(history, solve_time);
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

    std::vector<GlobalIndex> constrained_dofs;
    if (!constraints.empty()) {
        constrained_dofs.reserve(constraints.numConstraints());
        constraints.forEach([&constrained_dofs](const constraints::AffineConstraints::ConstraintView& cv) {
            if (cv.slave_dof >= 0) {
                constrained_dofs.push_back(cv.slave_dof);
            }
        });
    }

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

    auto computeResidualNorm = [&]() -> double {
        return residualNormForConvergence(r, residual_scratch);
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

        transient.system().beginTimeStep();
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_vector = true;
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
        return computeResidualNorm();
    };

    auto assembleJacobianOnly = [&](const systems::SystemStateView& state) {
        auto J_view = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix) op='" + options_.jacobian_op + "'");
        }
        transient.system().beginTimeStep();
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
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
        transient.system().beginTimeStep();
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
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
        transient.system().beginTimeStep();
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
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

    auto tolerancesSatisfied = [&](double norm) -> bool {
        const bool abs_ok = norm <= options_.abs_tolerance;
        const bool rel_ok = (options_.rel_tolerance <= 0.0)
            ? true
            : (report.residual_norm0 > 0.0
                   ? (norm / report.residual_norm0 <= options_.rel_tolerance)
                   : abs_ok);
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

        transient.system().beginTimeStep();
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

    for (int it = 0; it < max_it; ++it) {
        history.updateGhosts();

        if (!constraints.empty()) {
            constraints.distribute(history.u());
            history.u().updateGhosts();
        }

        const systems::SystemStateView state = makeStateView(history, solve_time);

        if (have_residual && !std::isfinite(current_residual_norm)) {
            // If the cached residual norm is invalid (e.g., NaN from a failed evaluation),
            // fall back to re-assembling the residual at the current state.
            have_residual = false;
        }

        const bool need_jacobian = !have_jacobian || (jacobian_period == 1) || ((it - last_jacobian_it) >= jacobian_period);
        bool jacobian_ready = have_jacobian && !need_jacobian;
        if (!have_residual) {
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
            have_residual = true;
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

        if (tolerancesSatisfied(current_residual_norm)) {
            report.converged = true;
            report.iterations = it;
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: converged before linear solve (tolerances satisfied).");
            }
            return report;
        }

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

                // Backup base state u.
                copyVector(u_backup, history.u());

                // Assemble r(u) with residual_op into residual_base.
                residual_base.zero();
                {
                    auto r_view = residual_base.createAssemblyView();
                    FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: jacobian check residual base view");

                    transient.system().beginTimeStep();
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    const auto ar = transient.assemble(req, state, nullptr, r_view.get());
                    FE_THROW_IF(!ar.success, FEException,
                                "NewtonSolver: Jacobian check base residual assembly failed: " + ar.error_message);
                }
                applyResidualFixups(residual_base);

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

                    transient.system().beginTimeStep();
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    const auto ar = transient.assemble(req, makeStateView(history, solve_time), nullptr, r_view.get());
                    FE_THROW_IF(!ar.success, FEException,
                                "NewtonSolver: Jacobian check perturbed residual assembly failed: " + ar.error_message);
                }
                applyResidualFixups(residual_scratch);

                // Restore u.
                copyVector(history.u(), u_backup);
                if (!constraints.empty()) {
                    constraints.distribute(history.u());
                }
                history.u().updateGhosts();

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

        int ptc_retries = 0;
        while (true) {
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: calling linear.solve()");
            }
            report.linear = linear.solve(J, du, r);
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

        if (options_.scale_dt_increments && !workspace.dt_field_dofs.empty()) {
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
        }

        const double du_norm = du.norm();

        if (!options_.use_line_search) {
            axpy(history.u(), static_cast<Real>(-1.0), du);
            if (!constraints.empty()) {
                constraints.distribute(history.u());
            }
            history.u().updateGhosts();
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
                    return report;
                }
            }
            continue;
        }

        // Backtracking line search: choose alpha in (0,1] so the residual norm decreases.
        copyVector(u_backup, history.u());
        const double r_norm0 = current_residual_norm;
        const double r_norm0_sq = r_norm0 * r_norm0;

        double alpha = 1.0;
        double alpha_last = alpha;
        double trial_norm = std::numeric_limits<double>::infinity();
        bool accepted = false;

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: line search begin alpha=1"
                << " alpha_min=" << options_.line_search_alpha_min
                << " shrink=" << options_.line_search_shrink
                << " c1=" << options_.line_search_c1;
            traceLog(oss.str());
        }

        for (int ls = 0; ls < options_.line_search_max_iterations; ++ls) {
            alpha_last = alpha;
            copyVector(history.u(), u_backup);
            axpy(history.u(), static_cast<Real>(-alpha), du);
            if (!constraints.empty()) {
                constraints.distribute(history.u());
            }
            history.u().updateGhosts();

            const systems::SystemStateView trial_state = makeStateView(history, solve_time);
            trial_norm = assembleResidualOnly(trial_state, /*phase=*/"line_search");

            bool ok = false;
            if (std::isfinite(trial_norm) && std::isfinite(r_norm0)) {
                // Armijo on phi(u) = 0.5*||r(u)||^2 with Newton direction.
                const double rhs = (1.0 - 2.0 * options_.line_search_c1 * alpha) * r_norm0_sq;
                if (rhs > 0.0) {
                    ok = (trial_norm * trial_norm <= rhs);
                } else {
                    ok = (trial_norm <= r_norm0);
                }
            }

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: line search trial alpha=" << alpha
                    << " ||r(alpha)||=" << trial_norm
                    << " ok=" << (ok ? 1 : 0);
                traceLog(oss.str());
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
            // When the line search fails to satisfy the Armijo condition, accept the last evaluated
            // trial point. Ensure `alpha` matches that state (important for step-tolerance logic).
            alpha = alpha_last;
        }

        if (!accepted && oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: line search did not satisfy decrease; accepting alpha=" << alpha
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
    return report;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
