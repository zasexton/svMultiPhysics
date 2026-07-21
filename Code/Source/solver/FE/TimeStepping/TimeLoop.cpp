/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/TimeLoop.h"

#include "Backends/Interfaces/DofPermutation.h"
#include "Core/FEException.h"
#include "Math/FiniteDifference.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "Sparsity/SparsityPattern.h"
#include "Systems/SystemsExceptions.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/CollocationMethods.h"
#include "TimeStepping/MultiStageScheme.h"
#include "TimeStepping/NewmarkBeta.h"
#include "TimeStepping/TimeSteppingUtils.h"
#include "TimeStepping/VSVO_BDF_Controller.h"
#include "Core/Logger.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

namespace {

class StepCandidateRollbackGuard {
public:
    explicit StepCandidateRollbackGuard(std::function<void()> rollback)
        : rollback_(std::move(rollback))
    {
    }

    StepCandidateRollbackGuard(const StepCandidateRollbackGuard&) = delete;
    StepCandidateRollbackGuard& operator=(
        const StepCandidateRollbackGuard&) = delete;

    ~StepCandidateRollbackGuard()
    {
        if (!armed_) {
            return;
        }
        try {
            discard();
        } catch (...) {
        }
    }

    void arm() noexcept { armed_ = true; }
    [[nodiscard]] bool armed() const noexcept { return armed_; }

    void discard()
    {
        if (!armed_) {
            return;
        }
        armed_ = false;
        if (rollback_) {
            rollback_();
        }
    }

    void release() noexcept { armed_ = false; }

private:
    std::function<void()> rollback_{};
    bool armed_{false};
};

void updateGhostsAndDistributeHistory(const constraints::AffineConstraints& constraints,
                                      TimeHistory& history)
{
    history.updateGhosts();
    if (constraints.empty()) {
        return;
    }

    constraints.distribute(history.u());
    for (int k = 1; k <= history.historyDepth(); ++k) {
        constraints.distribute(history.uPrevK(k));
    }
}

std::vector<GlobalIndex> collectDirichletDofs(const constraints::AffineConstraints& constraints)
{
    std::vector<GlobalIndex> dirichlet_dofs;
    if (constraints.empty()) {
        return dirichlet_dofs;
    }

    dirichlet_dofs.reserve(constraints.numConstraints());
    constraints.forEach([&dirichlet_dofs](const constraints::AffineConstraints::ConstraintView& cv) {
        if (cv.slave_dof >= 0 && cv.isDirichlet()) {
            dirichlet_dofs.push_back(cv.slave_dof);
        }
    });
    std::sort(dirichlet_dofs.begin(), dirichlet_dofs.end());
    dirichlet_dofs.erase(std::unique(dirichlet_dofs.begin(), dirichlet_dofs.end()),
                         dirichlet_dofs.end());
    return dirichlet_dofs;
}

std::size_t setLinearDirichletDofs(backends::LinearSolver& linear,
                                   const constraints::AffineConstraints& constraints)
{
    const auto dirichlet_dofs = collectDirichletDofs(constraints);
    linear.setDirichletDofs(dirichlet_dofs);
    return dirichlet_dofs.size();
}

[[nodiscard]] bool initializationDiagnosticsEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_TIMELOOP_INITIALIZATION_DIAGNOSTICS");
        if (env == nullptr || env[0] == '\0') {
            return false;
        }
        const std::string value(env);
        return !(value == "0" || value == "false" || value == "False" ||
                 value == "off" || value == "OFF" || value == "no" ||
                 value == "NO");
    }();
    return enabled;
}

/// Legacy comparison switch: zero the rate history at constrained DOFs after
/// each generalized-alpha rate update instead of keeping the
/// constraint-consistent finite-difference rates (which carry the
/// inhomogeneity rate g_dot of time-dependent Dirichlet data).
[[nodiscard]] bool zeroConstrainedRatesRequested() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_ZERO_CONSTRAINED_RATES");
        if (env == nullptr || env[0] == '\0') {
            return false;
        }
        const std::string value(env);
        return !(value == "0" || value == "false" || value == "False" ||
                 value == "off" || value == "OFF" || value == "no" ||
                 value == "NO");
    }();
    return enabled;
}

/// Legacy comparison switch: skip the accepted-step re-distribution of
/// master-bearing (MPC) constraint state into the committed solution and rate
/// history (see the accepted-step block in run()). With the switch set, DOFs
/// that enter or leave interface-tracking MPC sets (e.g. small-cut
/// aggregation) keep their raw finite-difference rates, which carry the
/// free-vs-extension value jump scaled by 1/(gamma*dt).
[[nodiscard]] bool mpcAcceptedStateDistributeDisabled() noexcept
{
    static const bool disabled = [] {
        const char* env = std::getenv("SVMP_NO_MPC_STATE_DISTRIBUTE");
        if (env == nullptr || env[0] == '\0') {
            return false;
        }
        const std::string value(env);
        return !(value == "0" || value == "false" || value == "False" ||
                 value == "off" || value == "OFF" || value == "no" ||
                 value == "NO");
    }();
    return disabled;
}

void logInitializationSolveDiagnostics(const char* phase,
                                       const constraints::AffineConstraints& constraints,
                                       std::size_t dirichlet_dofs,
                                       const backends::GenericMatrix& matrix,
                                       const backends::GenericVector& rhs)
{
    if (!initializationDiagnosticsEnabled()) {
        return;
    }

    std::ostringstream oss;
    oss << "TimeLoop: initialization linear solve diagnostics"
        << " diagnostic=timeloop_initialization_linear_solve"
        << " phase='" << phase << "'"
        << " constraints=" << constraints.numConstraints()
        << " dirichlet_dofs=" << dirichlet_dofs
        << " matrix_rows=" << matrix.numRows()
        << " matrix_cols=" << matrix.numCols()
        << " rhs_norm=" << rhs.norm();
    FE_LOG_INFO(oss.str());
}

std::vector<GlobalIndex> collectOwnedExactZeroMassRows(
    const systems::FESystem& system,
    const backends::GenericMatrix& matrix,
    const systems::OperatorTag& op)
{
    std::vector<GlobalIndex> zero_rows;
    const auto owned_dofs =
        system.dofHandler().getPartition().locallyOwned().toVector();
    zero_rows.reserve(owned_dofs.size());

    const auto* distributed = system.distributedSparsityIfAvailable(op);
    const auto permutation = system.dofPermutation();
    const bool distributed_is_permuted =
        distributed != nullptr &&
        distributed->dofIndexing() ==
            sparsity::DistributedSparsityPattern::DofIndexing::NodalInterleaved;

    const auto backendToFe = [&](GlobalIndex backend_dof) {
        if (!distributed_is_permuted) {
            return backend_dof;
        }
        if (!permutation || backend_dof < 0 ||
            static_cast<std::size_t>(backend_dof) >=
                permutation->inverse.size()) {
            return INVALID_GLOBAL_INDEX;
        }
        return permutation->inverse[static_cast<std::size_t>(backend_dof)];
    };

    const auto feToBackend = [&](GlobalIndex fe_dof) {
        if (!distributed_is_permuted) {
            return fe_dof;
        }
        if (!permutation || fe_dof < 0 ||
            static_cast<std::size_t>(fe_dof) >=
                permutation->forward.size()) {
            return INVALID_GLOBAL_INDEX;
        }
        return permutation->forward[static_cast<std::size_t>(fe_dof)];
    };

    const auto rowHasNonzeroInBackendColumns =
        [&](GlobalIndex fe_row, std::span<const GlobalIndex> backend_cols) {
            for (const auto backend_col : backend_cols) {
                const auto fe_col = backendToFe(backend_col);
                if (fe_col < 0 || fe_col >= matrix.numCols()) {
                    continue;
                }
                // Exact comparison is intentional. Small-cut mass is physical,
                // and a non-finite entry is not an empty row to regularize away.
                if (matrix.getEntry(fe_row, fe_col) != Real{0.0}) {
                    return true;
                }
            }
            return false;
        };

    const auto rowIsExactlyEmpty = [&](GlobalIndex fe_row) {
        if (distributed != nullptr && distributed->isFinalized()) {
            const auto backend_row = feToBackend(fe_row);
            if (backend_row >= 0) {
                if (distributed->ownsRow(backend_row)) {
                    const auto local_row =
                        backend_row - distributed->ownedRows().first;

                    const auto local_diag_cols =
                        distributed->getRowDiagCols(local_row);
                    for (const auto local_col : local_diag_cols) {
                        const auto backend_col =
                            distributed->ownedCols().first + local_col;
                        const auto fe_col = backendToFe(backend_col);
                        if (fe_col >= 0 && fe_col < matrix.numCols() &&
                            matrix.getEntry(fe_row, fe_col) != Real{0.0}) {
                            return false;
                        }
                    }

                    const auto local_offdiag_cols =
                        distributed->getRowOffdiagCols(local_row);
                    for (const auto local_col : local_offdiag_cols) {
                        const auto backend_col =
                            distributed->ghostColToGlobal(local_col);
                        const auto fe_col = backendToFe(backend_col);
                        if (fe_col >= 0 && fe_col < matrix.numCols() &&
                            matrix.getEntry(fe_row, fe_col) != Real{0.0}) {
                            return false;
                        }
                    }
                    return true;
                }

                const auto local_ghost_row =
                    distributed->globalToGhostRow(backend_row);
                if (local_ghost_row >= 0) {
                    return !rowHasNonzeroInBackendColumns(
                        fe_row,
                        distributed->getGhostRowCols(local_ghost_row));
                }
            }
        } else {
            const auto& pattern = system.sparsity(op);
            if (pattern.isFinalized() && fe_row < pattern.numRows()) {
                for (const auto col : pattern.getRowSpan(fe_row)) {
                    if (col >= 0 && col < matrix.numCols() &&
                        matrix.getEntry(fe_row, col) != Real{0.0}) {
                        return false;
                    }
                }
                return true;
            }
        }

        // A distributed pattern may not retain a particular non-owned row.
        // The startup solve is performed only once, so prefer a complete
        // numerical scan over guessing from the diagonal in that rare case.
        for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
            if (matrix.getEntry(fe_row, col) != Real{0.0}) {
                return false;
            }
        }
        return true;
    };

    for (const auto dof : owned_dofs) {
        if (dof < 0 || dof >= matrix.numRows() || dof >= matrix.numCols()) {
            continue;
        }
        // A zero diagonal alone does not imply an algebraic row: mixed and
        // cross-field differential operators can have a valid mass equation
        // entirely in off-diagonal blocks. Regularize only a numerically empty
        // row, retaining every exact nonzero regardless of magnitude.
        if (matrix.getEntry(dof, dof) == Real{0.0} &&
            rowIsExactlyEmpty(dof)) {
            zero_rows.push_back(dof);
        }
    }
    return zero_rows;
}

void logFirstOrderRateInitializationResult(
    const backends::SolverReport& report,
    bool accepted,
    std::size_t exact_zero_mass_rows,
    Real rate_norm,
    std::string_view fallback_reason)
{
    std::ostringstream oss;
    oss << "TimeLoop: first-order rate initialization"
        << " diagnostic=timeloop_first_order_rate_initialization"
        << " method=selective_term_linearized_rate_solve"
        << " accepted=" << (accepted ? 1 : 0)
        << " exact_zero_mass_owned_rows_regularized=" << exact_zero_mass_rows
        << " linear_converged=" << (report.converged ? 1 : 0)
        << " linear_iterations=" << report.iterations
        << " initial_residual_norm=" << report.initial_residual_norm
        << " final_residual_norm=" << report.final_residual_norm
        << " relative_residual=" << report.relative_residual
        << " rate_norm=" << rate_norm
        << " fallback_reason='" << fallback_reason << "'";
    FE_LOG_INFO(oss.str());
}

void copyVector(backends::GenericVector& dst, const backends::GenericVector& src)
{
    auto d = dst.localSpan();
    auto s = src.localSpan();
    FE_CHECK_ARG(d.size() == s.size(), "TimeLoop: vector size mismatch");
    std::copy(s.begin(), s.end(), d.begin());
}

/**
 * Roll back optional rate vectors unless the enclosing step attempt commits.
 *
 * Generalized-alpha and structural schemes update uDot/uDDot while forming a
 * converged candidate.  Acceptance callbacks and adaptive controllers run
 * after that update and can still reject the candidate.  `u()` is reset from
 * accepted solution history on the next attempt, but rate vectors have no
 * equivalent history slot, so they require an explicit transaction guard.
 */
class AttemptRateStateGuard {
public:
    AttemptRateStateGuard(TimeHistory& history,
                          const backends::BackendFactory& factory,
                          TimeHistory::RateStateSnapshot& snapshot)
        : history_(history)
        , snapshot_(snapshot)
    {
        history_.snapshotRateState(snapshot_, factory);
    }

    AttemptRateStateGuard(const AttemptRateStateGuard&) = delete;
    AttemptRateStateGuard& operator=(const AttemptRateStateGuard&) = delete;

    ~AttemptRateStateGuard() noexcept
    {
        if (!committed_) {
            history_.restoreRateState(snapshot_);
        }
    }

    void commit() noexcept { committed_ = true; }

private:
    TimeHistory& history_;
    TimeHistory::RateStateSnapshot& snapshot_;
    bool committed_{false};
};

systems::SystemStateView makeAcceptedTimeStepStateView(const TimeHistory& history,
                                                       double solve_time)
{
    systems::SystemStateView state;
    state.time = solve_time;
    state.dt = history.dt();
    state.effective_dt = history.dt();
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

systems::SystemStateView makeRestoredTimeStepStateView(
    const TimeHistory& history,
    double accepted_time,
    double attempted_dt)
{
    systems::SystemStateView state;
    state.time = accepted_time;
    state.dt = history.dtPrev() > 0.0 ? history.dtPrev() : attempted_dt;
    state.effective_dt = state.dt;
    state.dt_prev = history.dtHistory().size() > 1u
                        ? history.dtHistory()[1]
                        : state.dt;
    state.u = history.uPrevSpan();
    state.u_prev = history.uPrev2Span();
    state.u_prev2 = history.historyDepth() >= 3
                        ? history.uPrevKSpan(3)
                        : history.uPrev2Span();
    state.u_vector = &history.uPrev();
    state.u_prev_vector = &history.uPrev2();
    state.u_prev2_vector = history.historyDepth() >= 3
                               ? &history.uPrevK(3)
                               : &history.uPrev2();
    state.dt_history = history.dtHistory();
    return state;
}

void zeroVectorEntries(std::span<const GlobalIndex> dofs, backends::GenericVector& vec)
{
    if (dofs.empty()) {
        return;
    }

    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "TimeLoop: zeroVectorEntries view");
    view->beginAssemblyPhase();
    view->zeroVectorEntries(dofs);
    view->finalizeAssembly();
}

void copyVectorEntries(std::span<const GlobalIndex> dofs, backends::GenericVector& dst, backends::GenericVector& src)
{
    if (dofs.empty()) {
        return;
    }

    auto dst_view = dst.createAssemblyView();
    auto src_view = src.createAssemblyView();
    FE_CHECK_NOT_NULL(dst_view.get(), "TimeLoop: copyVectorEntries dst view");
    FE_CHECK_NOT_NULL(src_view.get(), "TimeLoop: copyVectorEntries src view");

    dst_view->beginAssemblyPhase();
    for (const auto dof : dofs) {
        const auto value = src_view->getVectorEntry(dof);
        dst_view->addVectorEntry(dof, value, assembly::AddMode::Insert);
    }
    dst_view->finalizeAssembly();
}

std::vector<GlobalIndex> collectNonTimeDerivativeDofs(
    const systems::FESystem& system,
    const std::vector<FieldId>& dt_fields)
{
    std::vector<GlobalIndex> nondt_dofs;
    if (dt_fields.empty()) {
        return nondt_dofs;
    }

    const auto& fmap = system.fieldMap();
    std::vector<unsigned char> is_dt_field(fmap.numFields(), 0u);
    for (const auto fid : dt_fields) {
        if (fid == INVALID_FIELD_ID) {
            continue;
        }
        const auto& rec = system.fieldRecord(fid);
        const auto field_index = fmap.getFieldIndex(rec.name);
        if (field_index < 0) {
            continue;
        }
        is_dt_field[static_cast<std::size_t>(field_index)] = 1u;
    }

    nondt_dofs.reserve(static_cast<std::size_t>(system.dofHandler().getNumDofs()));
    for (std::size_t field_index = 0; field_index < fmap.numFields(); ++field_index) {
        if (is_dt_field[field_index] != 0u) {
            continue;
        }
        const auto range = fmap.getFieldDofRange(field_index);
        for (GlobalIndex dof = range.first; dof < range.second; ++dof) {
            nondt_dofs.push_back(dof);
        }
    }
    return nondt_dofs;
}

class Dt1NoHistoryIntegrator final : public systems::TimeIntegrator {
public:
    [[nodiscard]] std::string name() const override { return "Dt1NoHistory"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 1; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const override
    {
        assembly::TimeIntegrationContext ctx;
        ctx.integrator_name = name();

        if (max_time_derivative_order <= 0) {
            return ctx;
        }
        FE_THROW_IF(max_time_derivative_order > maxSupportedDerivativeOrder(),
                    InvalidArgumentException,
                    "TimeIntegrator '" + name() + "' does not support dt(·," + std::to_string(max_time_derivative_order) + ")");

        const double dt = state.dt;
        FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), InvalidArgumentException,
                    "TimeIntegrator '" + name() + "': dt must be finite and > 0");

        assembly::TimeDerivativeStencil s;
        s.order = 1;
        s.a.assign(1, static_cast<Real>(1.0 / dt));
        ctx.dt1 = s;
        return ctx;
    }
};

class Dt12NoHistoryIntegrator final : public systems::TimeIntegrator {
public:
    [[nodiscard]] std::string name() const override { return "Dt12NoHistory"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const override
    {
        assembly::TimeIntegrationContext ctx;
        ctx.integrator_name = name();

        if (max_time_derivative_order <= 0) {
            return ctx;
        }
        FE_THROW_IF(max_time_derivative_order > maxSupportedDerivativeOrder(),
                    InvalidArgumentException,
                    "TimeIntegrator '" + name() + "' does not support dt(·," + std::to_string(max_time_derivative_order) + ")");

        const double dt = state.dt;
        FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), InvalidArgumentException,
                    "TimeIntegrator '" + name() + "': dt must be finite and > 0");

        if (max_time_derivative_order >= 1) {
            assembly::TimeDerivativeStencil s;
            s.order = 1;
            s.a.assign(1, static_cast<Real>(1.0 / dt));
            ctx.dt1 = s;
        }

        if (max_time_derivative_order >= 2) {
            assembly::TimeDerivativeStencil s;
            s.order = 2;
            s.a.assign(1, static_cast<Real>(1.0 / (dt * dt)));
            ctx.dt2 = s;
        }

        return ctx;
    }
};

class DerivativeWeightedIntegrator final : public systems::TimeIntegrator {
public:
    DerivativeWeightedIntegrator(std::shared_ptr<const systems::TimeIntegrator> base,
                                 Real time_derivative_weight,
                                 Real non_time_derivative_weight,
                                 Real dt1_term_weight,
                                 Real dt2_term_weight)
        : base_(std::move(base))
        , time_derivative_weight_(time_derivative_weight)
        , non_time_derivative_weight_(non_time_derivative_weight)
        , dt1_term_weight_(dt1_term_weight)
        , dt2_term_weight_(dt2_term_weight)
    {
        FE_CHECK_NOT_NULL(base_.get(), "DerivativeWeightedIntegrator::base");
    }

    [[nodiscard]] std::string name() const override { return base_->name(); }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return base_->maxSupportedDerivativeOrder(); }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const override
    {
        auto ctx = base_->buildContext(max_time_derivative_order, state);
        ctx.time_derivative_term_weight = time_derivative_weight_;
        ctx.non_time_derivative_term_weight = non_time_derivative_weight_;
        ctx.dt1_term_weight = dt1_term_weight_;
        ctx.dt2_term_weight = dt2_term_weight_;
        return ctx;
    }

private:
    std::shared_ptr<const systems::TimeIntegrator> base_{};
    Real time_derivative_weight_{1.0};
    Real non_time_derivative_weight_{1.0};
    Real dt1_term_weight_{1.0};
    Real dt2_term_weight_{1.0};
};

} // namespace

TimeLoop::TimeLoop(TimeLoopOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(!(options_.dt > 0.0) || !std::isfinite(options_.dt),
                InvalidArgumentException,
                "TimeLoop: dt must be finite and > 0");
    FE_THROW_IF(!(options_.t_end >= options_.t0) || !std::isfinite(options_.t0) || !std::isfinite(options_.t_end),
                InvalidArgumentException,
                "TimeLoop: t0/t_end must be finite and t_end >= t0");
    FE_THROW_IF(options_.max_steps <= 0, InvalidArgumentException,
                "TimeLoop: max_steps must be > 0");
    FE_THROW_IF(options_.last_step_absorb_fraction < 0.0 ||
                    !std::isfinite(options_.last_step_absorb_fraction),
                InvalidArgumentException,
                "TimeLoop: last_step_absorb_fraction must be finite and >= 0");

    if (options_.scheme == SchemeKind::ThetaMethod) {
        FE_THROW_IF(!(options_.theta >= 0.0 && options_.theta <= 1.0) || !std::isfinite(options_.theta),
                    InvalidArgumentException,
                    "TimeLoop: theta must be finite and in [0,1]");
    }
    if (options_.scheme == SchemeKind::TRBDF2) {
        FE_THROW_IF(!(options_.trbdf2_gamma > 0.0 && options_.trbdf2_gamma < 1.0) || !std::isfinite(options_.trbdf2_gamma),
                    InvalidArgumentException,
                    "TimeLoop: trbdf2_gamma must be finite and in (0,1)");
    }
    if (options_.scheme == SchemeKind::GeneralizedAlpha) {
        (void)utils::generalizedAlphaFirstOrderFromRhoInf(options_.generalized_alpha_rho_inf);
    }
    if (options_.scheme == SchemeKind::Newmark) {
        FE_THROW_IF(!(options_.newmark_beta > 0.0) || !std::isfinite(options_.newmark_beta),
                    InvalidArgumentException,
                    "TimeLoop: newmark_beta must be finite and > 0");
        FE_THROW_IF(!(options_.newmark_gamma > 0.0) || !std::isfinite(options_.newmark_gamma),
                    InvalidArgumentException,
                    "TimeLoop: newmark_gamma must be finite and > 0");
    }
    if (options_.scheme == SchemeKind::DG) {
        FE_THROW_IF(options_.dg_degree < 0, InvalidArgumentException,
                    "TimeLoop: dg_degree must be >= 0");
        FE_THROW_IF(options_.dg_degree > 10, InvalidArgumentException,
                    "TimeLoop: dg_degree too large (max 10)");
    }
    if (options_.scheme == SchemeKind::CG) {
        FE_THROW_IF(options_.cg_degree < 1, InvalidArgumentException,
                    "TimeLoop: cg_degree must be >= 1");
        FE_THROW_IF(options_.cg_degree > 10, InvalidArgumentException,
                    "TimeLoop: cg_degree too large (max 10)");
    }
    if (options_.scheme == SchemeKind::DG1 || options_.scheme == SchemeKind::DG ||
        options_.scheme == SchemeKind::CG2 || options_.scheme == SchemeKind::CG) {
        FE_THROW_IF(options_.collocation_max_outer_iterations <= 0, InvalidArgumentException,
                    "TimeLoop: collocation_max_outer_iterations must be > 0");
        FE_THROW_IF(options_.collocation_outer_tolerance < 0.0 || !std::isfinite(options_.collocation_outer_tolerance),
                    InvalidArgumentException,
                    "TimeLoop: collocation_outer_tolerance must be finite and >= 0");
    }
    if (options_.scheme == SchemeKind::VSVO_BDF) {
        FE_THROW_IF(!options_.step_controller, InvalidArgumentException,
                    "TimeLoop: VSVO_BDF requires a step_controller");
    }
}

TimeLoopReport TimeLoop::run(systems::TransientSystem& transient,
                             const backends::BackendFactory& factory,
                             backends::LinearSolver& linear,
                             TimeHistory& history,
                             const TimeLoopCallbacks& callbacks) const
{
    TimeLoopReport report;

    auto bdf1 = std::make_shared<systems::BackwardDifferenceIntegrator>();
    auto bdf2 = std::make_shared<systems::BDF2Integrator>();
    std::optional<utils::GeneralizedAlphaFirstOrderParams> ga1_params;
    std::optional<utils::GeneralizedAlphaSecondOrderParams> ga2_params;
    std::shared_ptr<const GeneralizedAlphaFirstOrderIntegrator> generalized_alpha_fo;
    std::shared_ptr<const GeneralizedAlphaSecondOrderIntegrator> generalized_alpha_so;
    std::shared_ptr<const NewmarkBetaIntegrator> newmark_beta;
    if (options_.scheme == SchemeKind::GeneralizedAlpha) {
        ga1_params = utils::generalizedAlphaFirstOrderFromRhoInf(options_.generalized_alpha_rho_inf);
        generalized_alpha_fo = std::make_shared<const GeneralizedAlphaFirstOrderIntegrator>(
            GeneralizedAlphaFirstOrderIntegratorOptions{
                .alpha_m = ga1_params->alpha_m,
                .alpha_f = ga1_params->alpha_f,
                .gamma = ga1_params->gamma,
                .history_rate_order = 0});
    }
    if (options_.scheme == SchemeKind::Newmark) {
        newmark_beta = std::make_shared<const NewmarkBetaIntegrator>(NewmarkBetaIntegratorOptions{
            .beta = options_.newmark_beta,
            .gamma = options_.newmark_gamma,
        });
    }

    const auto n_dofs = transient.system().dofHandler().getNumDofs();
    FE_THROW_IF(n_dofs <= 0, systems::InvalidStateException, "TimeLoop: system has no DOFs");

    const double t0 = options_.t0;
    const double t_end = options_.t_end;
    history.setTime(t0);
    history.setDt(options_.dt);
    if (!(history.dtPrev() > 0.0)) {
        history.setPrevDt(options_.dt);
    }

    NewtonSolver newton(options_.newton);
    NewtonWorkspace workspace;
    newton.allocateWorkspace(transient.system(), factory, workspace);
    MultiStageSolver stages(newton);

    // Interface-tracking constraints can change their slave/master topology
    // between steps, which re-augments the system sparsity patterns. The
    // workspace Jacobian was allocated from the setup-time pattern; backends
    // with immutable stored patterns (Eigen CSR) silently drop writes outside
    // it, so the matrix must be reallocated whenever the pattern revision moves.
    std::uint64_t workspace_sparsity_revision = transient.system().sparsityPatternRevision();
    auto ensure_workspace_matches_sparsity = [&]() {
        const auto revision = transient.system().sparsityPatternRevision();
        if (revision == workspace_sparsity_revision) {
            return;
        }
        newton.allocateWorkspace(transient.system(), factory, workspace);
        workspace_sparsity_revision = revision;
    };

    // Ensure time-history vectors use the same backend layout as the solver workspace.
    // For backends like FSILS, vectors created before any matrix exists may not share
    // the matrix's internal ordering, which would corrupt updates like u <- u - du.
    history.repack(factory);
    linear.setOptions(transient.system().augmentSolverOptions(linear.getOptions()));

    if (options_.scheme == SchemeKind::VSVO_BDF && history.stepIndex() > 0) {
        // Restart sanity check: variable-step schemes require dtHistory() to match the provided
        // displacement history. Avoid silently fabricating older dt values via primeDtHistory().
        const int required_dt_history = std::min(history.stepIndex(), history.historyDepth());
        FE_THROW_IF(!history.dtHistoryIsValid(required_dt_history),
                    InvalidArgumentException,
                    "TimeLoop: VSVO_BDF restart requires a consistent dtHistory() (use TimeHistory::setDtHistory)");
    }
    history.primeDtHistory(history.dtPrev() > 0.0 ? history.dtPrev() : history.dt());

    // Ensure the initial time-history states satisfy strong constraints (Dirichlet, etc.).
    //
    // Many transient formulations include dt(u) terms that use uPrev/uPrev2. If the history
    // states do not satisfy inhomogeneous Dirichlet data at t0 (e.g., nonzero inlet velocity),
    // the first nonlinear solve can start from an inconsistent dt(u) jump and fail to converge.
    //
    // For fresh runs (stepIndex==0), treat all allocated history states as the initial condition
    // at t0 and distribute constraints accordingly.
    if (history.stepIndex() == 0) {
        auto& sys = transient.system();
        const auto& constraints = sys.constraints();
        if (!constraints.empty()) {
            sys.updateConstraints(t0, history.dt());
            updateGhostsAndDistributeHistory(constraints, history);
        }
    }
    setLinearDirichletDofs(linear, transient.system().constraints());

    auto scratch_vec0 = factory.createVector(n_dofs);
    auto scratch_vec1 = factory.createVector(n_dofs);
    auto scratch_vec2 = factory.createVector(n_dofs);
    auto generalized_alpha_rate_n = factory.createVector(n_dofs);
    FE_CHECK_NOT_NULL(scratch_vec0.get(), "TimeLoop scratch_vec0");
    FE_CHECK_NOT_NULL(scratch_vec1.get(), "TimeLoop scratch_vec1");
    FE_CHECK_NOT_NULL(scratch_vec2.get(), "TimeLoop scratch_vec2");
    FE_CHECK_NOT_NULL(generalized_alpha_rate_n.get(),
                      "TimeLoop generalized-alpha rate_n scratch");

    auto dt12_nohistory = std::make_shared<const Dt12NoHistoryIntegrator>();

    auto ensureSecondOrderKinematics = [&](bool overwrite_u_dot, bool overwrite_u_ddot, bool require_u_ddot) {
        if (!overwrite_u_dot && !overwrite_u_ddot) {
            return;
        }

        history.ensureSecondOrderState(factory);

        const auto init = utils::initializeSecondOrderStateFromDisplacementHistory(
            history,
            history.uDot().localSpan(),
            history.uDDot().localSpan(),
            /*overwrite_u_dot=*/overwrite_u_dot,
            /*overwrite_u_ddot=*/overwrite_u_ddot);
        bool acceleration_initialized = init.initialized_acceleration;

        auto& sys = transient.system();
        const auto& constraints = sys.constraints();

        if (overwrite_u_ddot && !acceleration_initialized) {
            // Fall back to a residual-based acceleration initialization at the current time:
            //   M a_n + other(u_n, v_n, t_n) = 0  =>  M a_n = -other
            //
            // This is intended as a robust restart path when only displacement (and optionally velocity)
            // history is available.
            const double dt = (history.dtPrev() > 0.0 && std::isfinite(history.dtPrev()))
                ? history.dtPrev()
                : history.dt();
            FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), systems::InvalidStateException,
                        "TimeLoop: cannot initialize (uDot,uDDot) with invalid dt");

            FE_THROW_IF(!workspace.isAllocated(), systems::InvalidStateException,
                        "TimeLoop: Newton workspace not allocated for (uDot,uDDot) initialization");
            FE_CHECK_NOT_NULL(dt12_nohistory.get(), "TimeLoop: dt12_nohistory integrator");

            auto& mass = *workspace.jacobian;
            auto& rhs = *workspace.residual;
            auto& u_prev_scratch = *workspace.delta;

            const auto u_n = history.uPrevSpan();
            const auto v_n = history.uDotSpan();
            auto u_prev = u_prev_scratch.localSpan();
            FE_CHECK_ARG(u_prev.size() == u_n.size() && v_n.size() == u_n.size(),
                         "TimeLoop: size mismatch in (uDot,uDDot) initialization");

            for (std::size_t i = 0; i < u_prev.size(); ++i) {
                u_prev[i] = u_n[i] - static_cast<Real>(dt) * v_n[i];
            }
            if (!constraints.empty()) {
                constraints.distributeHomogeneous(u_prev_scratch);
            }

            rhs.zero();
            auto rhs_view = rhs.createAssemblyView();
            FE_CHECK_NOT_NULL(rhs_view.get(), "TimeLoop: initialization rhs assembly view");

            systems::SystemStateView state;
            state.time = history.time();
            state.dt = dt;
            state.dt_prev = history.dtPrev();
            state.u = u_n;
            state.u_prev = std::span<const Real>(u_prev.data(), u_prev.size());
            state.u_prev2 = u_n; // satisfies validation; dt2 terms are disabled below.

            auto other_integrator = std::make_shared<const DerivativeWeightedIntegrator>(
                bdf1,
                /*time_derivative_weight=*/static_cast<Real>(1.0),
                /*non_time_derivative_weight=*/static_cast<Real>(1.0),
                /*dt1_term_weight=*/static_cast<Real>(1.0),
                /*dt2_term_weight=*/static_cast<Real>(0.0));
            systems::TransientSystem transient_other(sys, other_integrator);
            systems::AssemblyRequest req_other;
            req_other.op = options_.newton.residual_op;
            req_other.want_vector = true;
            req_other.zero_outputs = true;

            transient_other.system().beginTimeStep();
            (void)transient_other.assemble(req_other, state, nullptr, rhs_view.get());

            auto b = rhs.localSpan();
            for (auto& v : b) {
                v = -v;
            }

            mass.zero();
            auto mass_view = mass.createAssemblyView();
            FE_CHECK_NOT_NULL(mass_view.get(), "TimeLoop: initialization mass assembly view");

            auto mass_integrator = std::make_shared<const DerivativeWeightedIntegrator>(
                dt12_nohistory,
                /*time_derivative_weight=*/static_cast<Real>(1.0),
                /*non_time_derivative_weight=*/static_cast<Real>(0.0),
                /*dt1_term_weight=*/static_cast<Real>(0.0),
                // Dt12NoHistory provides dt(u,2) with coeff 1/dt^2; scale so dt(u,2) is interpreted as "a".
                /*dt2_term_weight=*/static_cast<Real>(dt * dt));
            systems::TransientSystem transient_mass(sys, mass_integrator);
            systems::AssemblyRequest req_mass;
            req_mass.op = options_.newton.jacobian_op;
            req_mass.want_matrix = true;
            req_mass.zero_outputs = true;

            systems::SystemStateView state_mass;
            state_mass.time = history.time();
            state_mass.dt = dt;
            state_mass.dt_prev = history.dtPrev();
            state_mass.u = u_n;

            transient_mass.system().beginTimeStep();
            (void)transient_mass.assemble(req_mass, state_mass, mass_view.get(), nullptr);

            history.uDDot().zero();
            const auto init_dirichlet_dofs = setLinearDirichletDofs(linear, constraints);
            logInitializationSolveDiagnostics("u_ddot", constraints, init_dirichlet_dofs, mass, rhs);
            const auto solve_rep = linear.solve(mass, history.uDDot(), rhs);
            if (!solve_rep.converged) {
                if (require_u_ddot) {
                    FE_THROW(systems::InvalidStateException,
                             "TimeLoop: failed to initialize uDDot from residual (linear solve did not converge)");
                }
                history.uDDot().zero();
            } else {
                acceleration_initialized = true;
            }
        }

        if (!constraints.empty()) {
            if (overwrite_u_dot) {
                constraints.distributeHomogeneous(history.uDot());
            }
            if (overwrite_u_ddot) {
                constraints.distributeHomogeneous(history.uDDot());
            }
        }

        if (require_u_ddot && overwrite_u_ddot && !acceleration_initialized) {
            // If we got here, the residual-based fallback did not throw but also didn't
            // manage to establish a usable acceleration. Do not proceed silently.
            FE_THROW(systems::InvalidStateException,
                     "TimeLoop: missing initial uDDot for 2nd-order scheme; provide TimeHistory::uDDot or displacement history >= 3");
        }
    };

    auto solveThetaStep = [&](double theta, double solve_time, double dt) -> NewtonReport {
        ImplicitStageSpec stage;
        stage.integrator = bdf1;
        stage.weights.time_derivative = static_cast<Real>(1.0);
        stage.weights.non_time_derivative = static_cast<Real>(theta);
        stage.solve_time = solve_time;

        ResidualAdditionSpec add;
        add.integrator = bdf1;
        add.weights.time_derivative = static_cast<Real>(0.0);
        add.weights.non_time_derivative = static_cast<Real>(1.0 - theta);

        systems::SystemStateView prev_state;
        prev_state.time = history.time();
        prev_state.dt = dt;
        prev_state.dt_prev = history.dtPrev();
        prev_state.u = history.uPrevSpan();
        prev_state.u_prev = history.uPrevSpan();
        prev_state.u_prev2 = history.uPrev2Span();
        prev_state.u_history = history.uHistorySpans();
        prev_state.dt_history = history.dtHistory();

        add.state = prev_state;
        stage.residual_addition = add;

        return stages.solveImplicitStage(transient.system(), linear, history, workspace, stage, scratch_vec0.get());
    };

    auto writeStructuralHistoryConstants = [](double alpha_m,
                                             double alpha_f,
                                             double beta,
                                             double gamma,
                                             double dt,
                                             std::span<const Real> u_n,
                                             std::span<const Real> v_n,
                                             std::span<const Real> a_n,
                                             std::span<Real> out_const_a,
                                             std::span<Real> out_const_v) {
        FE_THROW_IF(!(alpha_f > 0.0) || !std::isfinite(alpha_f), InvalidArgumentException,
                    "TimeLoop: structural scheme requires finite alpha_f > 0");
        FE_THROW_IF(!(beta > 0.0) || !std::isfinite(beta), InvalidArgumentException,
                    "TimeLoop: structural scheme requires finite beta > 0");
        FE_THROW_IF(!(gamma > 0.0) || !std::isfinite(gamma), InvalidArgumentException,
                    "TimeLoop: structural scheme requires finite gamma > 0");
        FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), InvalidArgumentException,
                    "TimeLoop: structural scheme requires finite dt > 0");
        FE_CHECK_ARG(u_n.size() == v_n.size() && u_n.size() == a_n.size(), "TimeLoop: structural state size mismatch");
        FE_CHECK_ARG(out_const_a.size() == u_n.size() && out_const_v.size() == u_n.size(),
                     "TimeLoop: structural constants size mismatch");

        const double inv_dt = 1.0 / dt;
        const double inv_beta = 1.0 / beta;
        const double inv_beta_dt = inv_beta * inv_dt;
        const double inv_beta_dt2 = inv_beta_dt * inv_dt;

        const double a_c_u = -(alpha_m / alpha_f) * inv_beta_dt2;
        const double a_c_v = -alpha_m * inv_beta_dt;
        const double a_c_a = 1.0 - alpha_m * (0.5 * inv_beta);

        const double v_c_u = -gamma * inv_beta_dt;
        const double v_c_v = 1.0 - alpha_f * gamma * inv_beta;
        const double v_c_a = alpha_f * dt * (1.0 - gamma * (0.5 * inv_beta));

        for (std::size_t i = 0; i < u_n.size(); ++i) {
            out_const_a[i] = static_cast<Real>(a_c_u) * u_n[i] + static_cast<Real>(a_c_v) * v_n[i] + static_cast<Real>(a_c_a) * a_n[i];
            out_const_v[i] = static_cast<Real>(v_c_u) * u_n[i] + static_cast<Real>(v_c_v) * v_n[i] + static_cast<Real>(v_c_a) * a_n[i];
        }
    };

    class OffsetSystemView final : public assembly::GlobalSystemView {
    public:
        OffsetSystemView(assembly::GlobalSystemView& inner, GlobalIndex row_offset, GlobalIndex col_offset)
            : inner_(&inner)
            , row_offset_(row_offset)
            , col_offset_(col_offset)
        {
        }

        void addMatrixEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> local_matrix,
                              assembly::AddMode mode) override
        {
            addMatrixEntries(dofs, dofs, local_matrix, mode);
        }

        void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                              std::span<const GlobalIndex> col_dofs,
                              std::span<const Real> local_matrix,
                              assembly::AddMode mode) override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            shifted_rows_.resize(row_dofs.size());
            shifted_cols_.resize(col_dofs.size());
            for (std::size_t i = 0; i < row_dofs.size(); ++i) {
                shifted_rows_[i] = row_dofs[i] + row_offset_;
            }
            for (std::size_t j = 0; j < col_dofs.size(); ++j) {
                shifted_cols_[j] = col_dofs[j] + col_offset_;
            }
            inner_->addMatrixEntries(shifted_rows_, shifted_cols_, local_matrix, mode);
        }

        void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            inner_->addMatrixEntry(row + row_offset_, col + col_offset_, value, mode);
        }

        void setDiagonal(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
        {
            FE_THROW_IF(dofs.size() != values.size(), InvalidArgumentException, "OffsetSystemView::setDiagonal: size mismatch");
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                setDiagonal(dofs[i], values[i]);
            }
        }

        void setDiagonal(GlobalIndex dof, Real value) override
        {
            addMatrixEntry(dof, dof, value, assembly::AddMode::Insert);
        }

        void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            shifted_rows_.resize(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                shifted_rows_[i] = rows[i] + row_offset_;
            }
            inner_->zeroRows(shifted_rows_, set_diagonal);
        }

        void addVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> local_vector,
                              assembly::AddMode mode) override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            shifted_rows_.resize(dofs.size());
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                shifted_rows_[i] = dofs[i] + row_offset_;
            }
            inner_->addVectorEntries(shifted_rows_, local_vector, mode);
        }

        void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            inner_->addVectorEntry(dof + row_offset_, value, mode);
        }

        void setVectorEntries(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            shifted_rows_.resize(dofs.size());
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                shifted_rows_[i] = dofs[i] + row_offset_;
            }
            inner_->setVectorEntries(shifted_rows_, values);
        }

        void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            shifted_rows_.resize(dofs.size());
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                shifted_rows_[i] = dofs[i] + row_offset_;
            }
            inner_->zeroVectorEntries(shifted_rows_);
        }

        void beginAssemblyPhase() override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            inner_->beginAssemblyPhase();
        }

        void endAssemblyPhase() override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            inner_->endAssemblyPhase();
        }

        void finalizeAssembly() override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            inner_->finalizeAssembly();
        }

        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override
        {
            return inner_ ? inner_->getPhase() : assembly::AssemblyPhase::NotStarted;
        }

        [[nodiscard]] bool hasMatrix() const noexcept override { return inner_ ? inner_->hasMatrix() : false; }
        [[nodiscard]] bool hasVector() const noexcept override { return inner_ ? inner_->hasVector() : false; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override { return inner_ ? inner_->numRows() : 0; }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return inner_ ? inner_->numCols() : 0; }
        [[nodiscard]] std::string backendName() const override { return inner_ ? inner_->backendName() : "<null>"; }

        void zero() override
        {
            FE_CHECK_NOT_NULL(inner_, "OffsetSystemView::inner");
            inner_->zero();
        }

	    private:
	        assembly::GlobalSystemView* inner_{nullptr};
	        GlobalIndex row_offset_{0};
	        GlobalIndex col_offset_{0};
	        std::vector<GlobalIndex> shifted_rows_{};
	        std::vector<GlobalIndex> shifted_cols_{};
	    };

	    using CollocationMethod = collocation::CollocationMethod;
	    using CollocationFamily = collocation::CollocationFamily;
	    using SecondOrderCollocationData = collocation::SecondOrderCollocationData;

	    std::unordered_map<int, CollocationMethod> collocation_gauss{};
	    std::unordered_map<int, CollocationMethod> collocation_radau{};

	    auto getCollocationMethod = [&](CollocationFamily family, int stages) -> const CollocationMethod& {
	        auto& cache = (family == CollocationFamily::Gauss) ? collocation_gauss : collocation_radau;
	        auto it = cache.find(stages);
	        if (it != cache.end()) {
	            return it->second;
	        }
	        auto [ins_it, inserted] = cache.emplace(stages, collocation::buildCollocationMethod(family, stages));
	        FE_CHECK_ARG(inserted, "TimeLoop: failed to cache collocation method");
	        return ins_it->second;
	    };

	    std::unordered_map<int, SecondOrderCollocationData> collocation_so_gauss{};
	    std::unordered_map<int, SecondOrderCollocationData> collocation_so_radau{};

	    auto getSecondOrderCollocationData = [&](CollocationFamily family, int stages) -> const SecondOrderCollocationData& {
	        auto& cache = (family == CollocationFamily::Gauss) ? collocation_so_gauss : collocation_so_radau;
	        auto it = cache.find(stages);
	        if (it != cache.end()) {
	            return it->second;
	        }
	        const auto& method = getCollocationMethod(family, stages);
	        auto [ins_it, inserted] = cache.emplace(stages, collocation::buildSecondOrderCollocationData(method));
	        FE_CHECK_ARG(inserted, "TimeLoop: failed to cache collocation second-order data");
	        return ins_it->second;
	    };

    struct CollocationWorkspace {
        int stages{0};

        std::unique_ptr<backends::GenericMatrix> jacobian{};
        std::unique_ptr<backends::GenericVector> residual{};
        std::unique_ptr<backends::GenericVector> delta{};
        std::unique_ptr<backends::GenericVector> stage_values{};   // concatenated U (size stages*n_dofs)
        std::unique_ptr<backends::GenericVector> stage_combination{}; // scratch (size n_dofs)
        std::unique_ptr<backends::GenericVector> dv0{}; // scratch dt*v_n (size n_dofs)

        std::shared_ptr<const systems::TimeIntegrator> dt_integrator{};
    };

    CollocationWorkspace collocation{};

    auto ensureCollocationWorkspace = [&](int stages_needed, bool need_block_system) {
        const bool stage_ok =
            collocation.stage_values && collocation.stage_combination && collocation.dv0 &&
            collocation.stages == stages_needed;
        const bool block_ok =
            collocation.jacobian && collocation.residual && collocation.delta;

        if (stage_ok && (!need_block_system || block_ok)) {
            return;
        }

        CollocationWorkspace next{};
        next.stages = stages_needed;
        next.dt_integrator = std::make_shared<const Dt12NoHistoryIntegrator>();
        next.stage_values = factory.createVector(static_cast<GlobalIndex>(stages_needed) * n_dofs);
        next.stage_combination = factory.createVector(n_dofs);
        next.dv0 = factory.createVector(n_dofs);

        FE_CHECK_NOT_NULL(next.stage_values.get(), "TimeLoop: collocation stage_values");
        FE_CHECK_NOT_NULL(next.stage_combination.get(), "TimeLoop: collocation stage_combination");
        FE_CHECK_NOT_NULL(next.dv0.get(), "TimeLoop: collocation dv0");

        if (need_block_system) {
            const auto& base_pattern = transient.system().sparsity(options_.newton.jacobian_op);
            sparsity::SparsityPattern block_pattern(static_cast<GlobalIndex>(stages_needed) * n_dofs,
                                                    static_cast<GlobalIndex>(stages_needed) * n_dofs);

            for (int bi = 0; bi < stages_needed; ++bi) {
                const GlobalIndex row_offset = static_cast<GlobalIndex>(bi) * n_dofs;
                for (GlobalIndex r = 0; r < n_dofs; ++r) {
                    const auto cols = base_pattern.getRowSpan(r);
                    for (int bj = 0; bj < stages_needed; ++bj) {
                        const GlobalIndex col_offset = static_cast<GlobalIndex>(bj) * n_dofs;
                        for (const GlobalIndex c : cols) {
                            block_pattern.addEntry(row_offset + r, col_offset + c);
                        }
                    }
                }
            }
            block_pattern.finalize();

            next.jacobian = factory.createMatrix(block_pattern);
            next.residual = factory.createVector(static_cast<GlobalIndex>(stages_needed) * n_dofs);
            next.delta = factory.createVector(static_cast<GlobalIndex>(stages_needed) * n_dofs);

            FE_CHECK_NOT_NULL(next.jacobian.get(), "TimeLoop: collocation jacobian");
            FE_CHECK_NOT_NULL(next.residual.get(), "TimeLoop: collocation residual");
            FE_CHECK_NOT_NULL(next.delta.get(), "TimeLoop: collocation delta");

            next.jacobian->zero();
            next.residual->zero();
            next.delta->zero();
        }

        next.stage_values->zero();
        next.stage_combination->zero();
        next.dv0->zero();

        collocation = std::move(next);
    };

    auto solveCollocationStep = [&](const CollocationMethod& method,
                                    double t_step,
                                    double dt_step) -> NewtonReport {
        FE_THROW_IF(method.stages <= 0, InvalidArgumentException, "TimeLoop: invalid collocation method");
        FE_THROW_IF(static_cast<int>(method.c.size()) != method.stages ||
                        static_cast<int>(method.row_sums.size()) != method.stages ||
                        static_cast<int>(method.ainv.size()) != method.stages * method.stages,
                    InvalidArgumentException,
                    "TimeLoop: invalid collocation method coefficients");
        if (!method.stiffly_accurate) {
            FE_THROW_IF(static_cast<int>(method.final_w.size()) != method.stages, InvalidArgumentException,
                        "TimeLoop: collocation method requires final_w");
        }

        const int temporal_order = transient.system().temporalOrder();
        FE_THROW_IF(temporal_order != 1 && temporal_order != 2, NotImplementedException,
                    "TimeLoop: cG/dG collocation supports temporal order 1 (dt(u)) or 2 (dt(u,2))");

        const bool use_stage_gauss_seidel =
            (options_.collocation_solve == CollocationSolveStrategy::StageGaussSeidel);
        FE_THROW_IF(!use_stage_gauss_seidel &&
                        !options_.newton.field_residual_criteria.empty(),
                    InvalidArgumentException,
                    "TimeLoop: monolithic collocation does not support "
                    "Newton field_residual_criteria; use the "
                    "StageGaussSeidel collocation solve strategy");

        ensureCollocationWorkspace(method.stages, /*need_block_system=*/!use_stage_gauss_seidel);

        auto& sys = transient.system();
        const auto& constraints = sys.constraints();

        const CollocationFamily family = method.stiffly_accurate ? CollocationFamily::RadauIIA : CollocationFamily::Gauss;
        const SecondOrderCollocationData* so_data = nullptr;
        if (temporal_order == 2) {
            so_data = &getSecondOrderCollocationData(family, method.stages);
        }

        if (temporal_order == 2) {
            const bool had_u_dot = history.hasUDotState();
            ensureSecondOrderKinematics(/*overwrite_u_dot=*/!had_u_dot,
                                        /*overwrite_u_ddot=*/false,
                                        /*require_u_ddot=*/false);
        }

        updateGhostsAndDistributeHistory(constraints, history);
        const auto u_n = history.uPrevSpan();
        FE_CHECK_ARG(static_cast<GlobalIndex>(u_n.size()) == n_dofs, "TimeLoop: collocation u_n size mismatch");

        auto U_all = collocation.stage_values->localSpan();
        FE_CHECK_ARG(U_all.size() == static_cast<std::size_t>(method.stages) * static_cast<std::size_t>(n_dofs),
                     "TimeLoop: collocation stage_values size mismatch");

        // Initial guess: set all stages to u_n.
        for (int s = 0; s < method.stages; ++s) {
            auto Ui = U_all.subspan(static_cast<std::size_t>(s) * static_cast<std::size_t>(n_dofs),
                                    static_cast<std::size_t>(n_dofs));
            std::copy(u_n.begin(), u_n.end(), Ui.begin());
        }
        if (!constraints.empty()) {
            for (int s = 0; s < method.stages; ++s) {
                auto Ui = U_all.subspan(static_cast<std::size_t>(s) * static_cast<std::size_t>(n_dofs),
                                        static_cast<std::size_t>(n_dofs));
                constraints.distribute(reinterpret_cast<double*>(Ui.data()),
                                       static_cast<GlobalIndex>(Ui.size()));
            }
        }

        if (temporal_order == 2) {
            auto dv0 = collocation.dv0->localSpan();
            const auto v_n = history.uDotSpan();
            FE_CHECK_ARG(dv0.size() == v_n.size(), "TimeLoop: collocation dv0 size mismatch");
            for (std::size_t i = 0; i < dv0.size(); ++i) {
                dv0[i] = static_cast<Real>(dt_step) * v_n[i];
            }
        }

        NewtonReport rep;

        if (use_stage_gauss_seidel) {
            rep.converged = true;
            rep.iterations = 0;
            rep.residual_norm0 = 0.0;
            rep.residual_norm = 0.0;

            const int max_outer = options_.collocation_max_outer_iterations;
            const double user_tol = options_.collocation_outer_tolerance;
            const double u_scale = std::max(1.0, history.uPrev().norm());
            const double tiny_tol = 10.0 * std::numeric_limits<double>::epsilon() * u_scale;

            systems::AssemblyRequest req_dt;
            req_dt.op = options_.newton.residual_op;
            req_dt.want_vector = true;
            req_dt.zero_outputs = false; // we explicitly clear scratch vectors

            auto dt1_only = std::make_shared<const DerivativeWeightedIntegrator>(
                collocation.dt_integrator,
                /*time_derivative_weight=*/static_cast<Real>(1.0),
                /*non_time_derivative_weight=*/static_cast<Real>(0.0),
                /*dt1_term_weight=*/static_cast<Real>(1.0),
                /*dt2_term_weight=*/static_cast<Real>(0.0));
            auto transient_dt1 = std::make_unique<systems::TransientSystem>(sys, dt1_only);
            FE_CHECK_NOT_NULL(transient_dt1.get(), "TimeLoop: collocation transient_dt1");

            std::unique_ptr<systems::TransientSystem> transient_dt2{};
            if (temporal_order == 2) {
                FE_CHECK_NOT_NULL(so_data, "TimeLoop: collocation second-order data");
                auto dt2_only = std::make_shared<const DerivativeWeightedIntegrator>(
                    collocation.dt_integrator,
                    /*time_derivative_weight=*/static_cast<Real>(1.0),
                    /*non_time_derivative_weight=*/static_cast<Real>(0.0),
                    /*dt1_term_weight=*/static_cast<Real>(0.0),
                    /*dt2_term_weight=*/static_cast<Real>(1.0));
                transient_dt2 = std::make_unique<systems::TransientSystem>(sys, dt2_only);
                FE_CHECK_NOT_NULL(transient_dt2.get(), "TimeLoop: collocation transient_dt2");
            }

            auto stage_old = scratch_vec2->localSpan();
            FE_CHECK_ARG(stage_old.size() == u_n.size(), "TimeLoop: collocation stage_old size mismatch");

            double last_update = 0.0;
            for (int outer = 0; outer < max_outer; ++outer) {
                double max_update = 0.0;
                double max_res = 0.0;
                bool all_converged = true;

                for (int i = 0; i < method.stages; ++i) {
                    const double stage_time = t_step + method.c[static_cast<std::size_t>(i)] * dt_step;
                    auto Ui = U_all.subspan(static_cast<std::size_t>(i) * static_cast<std::size_t>(n_dofs),
                                            static_cast<std::size_t>(n_dofs));
                    FE_CHECK_ARG(Ui.size() == stage_old.size(), "TimeLoop: collocation stage size mismatch");

                    std::copy(Ui.begin(), Ui.end(), stage_old.begin());

                    auto u_guess = history.uSpan();
                    FE_CHECK_ARG(u_guess.size() == Ui.size(), "TimeLoop: collocation history.u size mismatch");
                    std::copy(Ui.begin(), Ui.end(), u_guess.begin());

                    scratch_vec0->zero();
                    auto add_view = scratch_vec0->createAssemblyView();
                    FE_CHECK_NOT_NULL(add_view.get(), "TimeLoop: collocation GS residual-add view");

                    systems::SystemStateView add_state;
                    add_state.time = stage_time;
                    add_state.dt = dt_step;
                    add_state.dt_prev = history.dtPrev();

                    Real dt1_coeff = static_cast<Real>(0.0);
                    Real dt2_coeff = static_cast<Real>(0.0);

                    auto w = collocation.stage_combination->localSpan();
                    if (temporal_order == 1) {
                        std::fill(w.begin(), w.end(), static_cast<Real>(0.0));
                        for (int j = 0; j < method.stages; ++j) {
                            if (j == i) continue;
                            const double a = method.ainv[static_cast<std::size_t>(i * method.stages + j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < w.size(); ++k) {
                                w[k] += static_cast<Real>(a) * Uj[k];
                            }
                        }
                        const double ssum = method.row_sums[static_cast<std::size_t>(i)];
                        for (std::size_t k = 0; k < w.size(); ++k) {
                            w[k] -= static_cast<Real>(ssum) * u_n[k];
                        }
                        if (!constraints.empty()) {
                            constraints.distributeHomogeneous(*collocation.stage_combination);
                        }

                        add_state.u = std::span<const Real>(w.data(), w.size());
                        transient_dt1->system().beginTimeStep();
                        (void)transient_dt1->assemble(req_dt, add_state, nullptr, add_view.get());

                        const double aii = method.ainv[static_cast<std::size_t>(i * method.stages + i)];
                        dt1_coeff = static_cast<Real>(aii);
                        dt2_coeff = static_cast<Real>(0.0);
                    } else {
                        FE_CHECK_NOT_NULL(so_data, "TimeLoop: collocation second-order data");
                        FE_CHECK_NOT_NULL(transient_dt2.get(), "TimeLoop: collocation transient_dt2");
                        const auto dv0 = collocation.dv0->localSpan();

                        // Constant part for dt(u).
                        std::fill(w.begin(), w.end(), static_cast<Real>(0.0));
                        const double c1_u0 = so_data->d1_u0[static_cast<std::size_t>(i)];
                        const double c1_dv0 = so_data->d1_dv0[static_cast<std::size_t>(i)];
                        for (std::size_t k = 0; k < w.size(); ++k) {
                            w[k] += static_cast<Real>(c1_u0) * u_n[k] + static_cast<Real>(c1_dv0) * dv0[k];
                        }
                        for (int j = 0; j < method.stages; ++j) {
                            if (j == i) continue;
                            const double a1 = so_data->d1[static_cast<std::size_t>(i * method.stages + j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < w.size(); ++k) {
                                w[k] += static_cast<Real>(a1) * Uj[k];
                            }
                        }
                        if (!constraints.empty()) {
                            constraints.distributeHomogeneous(*collocation.stage_combination);
                        }

                        add_state.u = std::span<const Real>(w.data(), w.size());
                        transient_dt1->system().beginTimeStep();
                        (void)transient_dt1->assemble(req_dt, add_state, nullptr, add_view.get());

                        // Constant part for dt(u,2) added into the same residual-add vector.
                        std::fill(w.begin(), w.end(), static_cast<Real>(0.0));
                        const double c2_u0 = so_data->d2_u0[static_cast<std::size_t>(i)];
                        const double c2_dv0 = so_data->d2_dv0[static_cast<std::size_t>(i)];
                        for (std::size_t k = 0; k < w.size(); ++k) {
                            w[k] += static_cast<Real>(c2_u0) * u_n[k] + static_cast<Real>(c2_dv0) * dv0[k];
                        }
                        for (int j = 0; j < method.stages; ++j) {
                            if (j == i) continue;
                            const double a2 = so_data->d2[static_cast<std::size_t>(i * method.stages + j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < w.size(); ++k) {
                                w[k] += static_cast<Real>(a2) * Uj[k];
                            }
                        }
                        if (!constraints.empty()) {
                            constraints.distributeHomogeneous(*collocation.stage_combination);
                        }

                        add_state.u = std::span<const Real>(w.data(), w.size());
                        transient_dt2->system().beginTimeStep();
                        (void)transient_dt2->assemble(req_dt, add_state, nullptr, add_view.get());

                        const double a1ii = so_data->d1[static_cast<std::size_t>(i * method.stages + i)];
                        const double a2ii = so_data->d2[static_cast<std::size_t>(i * method.stages + i)];
                        dt1_coeff = static_cast<Real>(a1ii);
                        dt2_coeff = static_cast<Real>(a2ii);
                    }

                    auto stage_integrator = std::make_shared<const DerivativeWeightedIntegrator>(
                        collocation.dt_integrator,
                        /*time_derivative_weight=*/static_cast<Real>(1.0),
                        /*non_time_derivative_weight=*/static_cast<Real>(1.0),
                        /*dt1_term_weight=*/dt1_coeff,
                        /*dt2_term_weight=*/dt2_coeff);
                    systems::TransientSystem transient_stage(sys, stage_integrator);
                    const auto nr_i = newton.solveStep(transient_stage, linear, stage_time, history, workspace, scratch_vec0.get());

                    rep.iterations += nr_i.iterations;
                    rep.linear = nr_i.linear;
                    max_res = std::max(max_res, nr_i.residual_norm);
                    if (outer == 0) {
                        rep.residual_norm0 = std::max(rep.residual_norm0, nr_i.residual_norm0);
                    }
                    all_converged = all_converged && nr_i.converged;

                    auto u_new = history.uSpan();
                    std::copy(u_new.begin(), u_new.end(), Ui.begin());

                    double diff2 = 0.0;
                    for (std::size_t k = 0; k < u_new.size(); ++k) {
                        const double diff = static_cast<double>(u_new[k] - stage_old[k]);
                        diff2 += diff * diff;
                    }
                    max_update = std::max(max_update, std::sqrt(diff2));
                }

                rep.residual_norm = max_res;
                last_update = max_update;

                if (!all_converged) {
                    rep.converged = false;
                    return rep;
                }

                const double tol = (user_tol > 0.0) ? user_tol : tiny_tol;
                if (max_update <= tol) {
                    break;
                }
            }

            if (user_tol > 0.0 && last_update > user_tol) {
                rep.converged = false;
                return rep;
            }
        } else {
            const int max_it = options_.newton.max_iterations;
            double prev_residual_norm = -1.0;
            for (int it = 0; it < max_it; ++it) {
            collocation.jacobian->zero();
            collocation.residual->zero();

            systems::AssemblyRequest req_matrix;
            req_matrix.op = options_.newton.jacobian_op;
            req_matrix.want_matrix = true;
            req_matrix.zero_outputs = false;

            systems::AssemblyRequest req_vector;
            req_vector.op = options_.newton.residual_op;
            req_vector.want_vector = true;
            req_vector.zero_outputs = false;

            for (int i = 0; i < method.stages; ++i) {
                const double stage_time = t_step + method.c[static_cast<std::size_t>(i)] * dt_step;
                const GlobalIndex row_offset = static_cast<GlobalIndex>(i) * n_dofs;

                auto Ui = U_all.subspan(static_cast<std::size_t>(i) * static_cast<std::size_t>(n_dofs),
                                        static_cast<std::size_t>(n_dofs));

                if (temporal_order == 1) {
                    // Assemble dt(u) term residual using the A^{-1} combination.
                    {
                        auto w = collocation.stage_combination->localSpan();
                        std::fill(w.begin(), w.end(), static_cast<Real>(0.0));

                        for (int j = 0; j < method.stages; ++j) {
                            const double a = method.ainv[static_cast<std::size_t>(i * method.stages + j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < Uj.size(); ++k) {
                                w[k] += static_cast<Real>(a) * Uj[k];
                            }
                        }

                        const double ssum = method.row_sums[static_cast<std::size_t>(i)];
                        for (std::size_t k = 0; k < w.size(); ++k) {
                            w[k] -= static_cast<Real>(ssum) * u_n[k];
                        }

                        auto dt_only = std::make_shared<const WeightedIntegrator>(collocation.dt_integrator,
                                                                                  /*time_derivative_weight=*/static_cast<Real>(1.0),
                                                                                  /*non_time_derivative_weight=*/static_cast<Real>(0.0));
                        systems::TransientSystem transient_dt(sys, dt_only);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(w.data(), w.size());

                        auto r_view = collocation.residual->createAssemblyView();
                        FE_CHECK_NOT_NULL(r_view.get(), "TimeLoop: collocation residual view");
                        OffsetSystemView r_block(*r_view, row_offset, /*col_offset=*/0);
                        sys.beginTimeStep();
                        (void)transient_dt.assemble(req_vector, state, nullptr, &r_block);
                    }

                    // Assemble non-dt residual at U_i.
                    {
                        auto non_dt = std::make_shared<const WeightedIntegrator>(collocation.dt_integrator,
                                                                                 /*time_derivative_weight=*/static_cast<Real>(0.0),
                                                                                 /*non_time_derivative_weight=*/static_cast<Real>(1.0));
                        systems::TransientSystem transient_nd(sys, non_dt);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(Ui.data(), Ui.size());

                        auto r_view = collocation.residual->createAssemblyView();
                        FE_CHECK_NOT_NULL(r_view.get(), "TimeLoop: collocation residual view");
                        OffsetSystemView r_block(*r_view, row_offset, /*col_offset=*/0);
                        sys.beginTimeStep();
                        (void)transient_nd.assemble(req_vector, state, nullptr, &r_block);
                    }

                    // Assemble Jacobian blocks.
                    for (int j = 0; j < method.stages; ++j) {
                        const GlobalIndex col_offset = static_cast<GlobalIndex>(j) * n_dofs;
                        const double a = method.ainv[static_cast<std::size_t>(i * method.stages + j)];

                        auto dt_block = std::make_shared<const WeightedIntegrator>(collocation.dt_integrator,
                                                                                   /*time_derivative_weight=*/static_cast<Real>(a),
                                                                                   /*non_time_derivative_weight=*/static_cast<Real>(0.0));
                        systems::TransientSystem transient_dt(sys, dt_block);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(Ui.data(), Ui.size());

                        auto J_view = collocation.jacobian->createAssemblyView();
                        FE_CHECK_NOT_NULL(J_view.get(), "TimeLoop: collocation jacobian view");
                        OffsetSystemView J_block(*J_view, row_offset, col_offset);
                        sys.beginTimeStep();
                        (void)transient_dt.assemble(req_matrix, state, &J_block, nullptr);
                    }

                    // Non-dt Jacobian (diagonal stage block only).
                    {
                        auto non_dt = std::make_shared<const WeightedIntegrator>(collocation.dt_integrator,
                                                                                 /*time_derivative_weight=*/static_cast<Real>(0.0),
                                                                                 /*non_time_derivative_weight=*/static_cast<Real>(1.0));
                        systems::TransientSystem transient_nd(sys, non_dt);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(Ui.data(), Ui.size());

                        auto J_view = collocation.jacobian->createAssemblyView();
                        FE_CHECK_NOT_NULL(J_view.get(), "TimeLoop: collocation jacobian view");
                        OffsetSystemView J_block(*J_view, row_offset, row_offset);
                        sys.beginTimeStep();
                        (void)transient_nd.assemble(req_matrix, state, &J_block, nullptr);
                    }
                } else {
                    FE_CHECK_NOT_NULL(so_data, "TimeLoop: collocation second-order data");
                    const auto dv0 = collocation.dv0->localSpan();

                    // Assemble dt(u) residual using Hermite stage derivatives.
                    {
                        auto w = collocation.stage_combination->localSpan();
                        std::fill(w.begin(), w.end(), static_cast<Real>(0.0));

                        const double c_u0 = so_data->d1_u0[static_cast<std::size_t>(i)];
                        const double c_dv0 = so_data->d1_dv0[static_cast<std::size_t>(i)];
                        for (std::size_t k = 0; k < w.size(); ++k) {
                            w[k] += static_cast<Real>(c_u0) * u_n[k] + static_cast<Real>(c_dv0) * dv0[k];
                        }
                        for (int j = 0; j < method.stages; ++j) {
                            const double a = so_data->d1[static_cast<std::size_t>(i * method.stages + j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < Uj.size(); ++k) {
                                w[k] += static_cast<Real>(a) * Uj[k];
                            }
                        }

                        auto dt1_only = std::make_shared<const DerivativeWeightedIntegrator>(
                            collocation.dt_integrator,
                            /*time_derivative_weight=*/static_cast<Real>(1.0),
                            /*non_time_derivative_weight=*/static_cast<Real>(0.0),
                            /*dt1_term_weight=*/static_cast<Real>(1.0),
                            /*dt2_term_weight=*/static_cast<Real>(0.0));
                        systems::TransientSystem transient_dt(sys, dt1_only);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(w.data(), w.size());

                        auto r_view = collocation.residual->createAssemblyView();
                        FE_CHECK_NOT_NULL(r_view.get(), "TimeLoop: collocation residual view");
                        OffsetSystemView r_block(*r_view, row_offset, /*col_offset=*/0);
                        sys.beginTimeStep();
                        (void)transient_dt.assemble(req_vector, state, nullptr, &r_block);
                    }

                    // Assemble dt(u,2) residual using Hermite stage derivatives.
                    {
                        auto w = collocation.stage_combination->localSpan();
                        std::fill(w.begin(), w.end(), static_cast<Real>(0.0));

                        const double c_u0 = so_data->d2_u0[static_cast<std::size_t>(i)];
                        const double c_dv0 = so_data->d2_dv0[static_cast<std::size_t>(i)];
                        for (std::size_t k = 0; k < w.size(); ++k) {
                            w[k] += static_cast<Real>(c_u0) * u_n[k] + static_cast<Real>(c_dv0) * dv0[k];
                        }
                        for (int j = 0; j < method.stages; ++j) {
                            const double a = so_data->d2[static_cast<std::size_t>(i * method.stages + j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < Uj.size(); ++k) {
                                w[k] += static_cast<Real>(a) * Uj[k];
                            }
                        }

                        auto dt2_only = std::make_shared<const DerivativeWeightedIntegrator>(
                            collocation.dt_integrator,
                            /*time_derivative_weight=*/static_cast<Real>(1.0),
                            /*non_time_derivative_weight=*/static_cast<Real>(0.0),
                            /*dt1_term_weight=*/static_cast<Real>(0.0),
                            /*dt2_term_weight=*/static_cast<Real>(1.0));
                        systems::TransientSystem transient_dt(sys, dt2_only);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(w.data(), w.size());

                        auto r_view = collocation.residual->createAssemblyView();
                        FE_CHECK_NOT_NULL(r_view.get(), "TimeLoop: collocation residual view");
                        OffsetSystemView r_block(*r_view, row_offset, /*col_offset=*/0);
                        sys.beginTimeStep();
                        (void)transient_dt.assemble(req_vector, state, nullptr, &r_block);
                    }

                    // Assemble non-dt residual at U_i.
                    {
                        auto non_dt = std::make_shared<const DerivativeWeightedIntegrator>(
                            collocation.dt_integrator,
                            /*time_derivative_weight=*/static_cast<Real>(0.0),
                            /*non_time_derivative_weight=*/static_cast<Real>(1.0),
                            /*dt1_term_weight=*/static_cast<Real>(0.0),
                            /*dt2_term_weight=*/static_cast<Real>(0.0));
                        systems::TransientSystem transient_nd(sys, non_dt);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(Ui.data(), Ui.size());

                        auto r_view = collocation.residual->createAssemblyView();
                        FE_CHECK_NOT_NULL(r_view.get(), "TimeLoop: collocation residual view");
                        OffsetSystemView r_block(*r_view, row_offset, /*col_offset=*/0);
                        sys.beginTimeStep();
                        (void)transient_nd.assemble(req_vector, state, nullptr, &r_block);
                    }

                    // Assemble Jacobian blocks (dt terms).
                    for (int j = 0; j < method.stages; ++j) {
                        const GlobalIndex col_offset = static_cast<GlobalIndex>(j) * n_dofs;
                        const double a1 = so_data->d1[static_cast<std::size_t>(i * method.stages + j)];
                        const double a2 = so_data->d2[static_cast<std::size_t>(i * method.stages + j)];

                        auto dt_block = std::make_shared<const DerivativeWeightedIntegrator>(
                            collocation.dt_integrator,
                            /*time_derivative_weight=*/static_cast<Real>(1.0),
                            /*non_time_derivative_weight=*/static_cast<Real>(0.0),
                            /*dt1_term_weight=*/static_cast<Real>(a1),
                            /*dt2_term_weight=*/static_cast<Real>(a2));
                        systems::TransientSystem transient_dt(sys, dt_block);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(Ui.data(), Ui.size());

                        auto J_view = collocation.jacobian->createAssemblyView();
                        FE_CHECK_NOT_NULL(J_view.get(), "TimeLoop: collocation jacobian view");
                        OffsetSystemView J_block(*J_view, row_offset, col_offset);
                        sys.beginTimeStep();
                        (void)transient_dt.assemble(req_matrix, state, &J_block, nullptr);
                    }

                    // Non-dt Jacobian (diagonal stage block only).
                    {
                        auto non_dt = std::make_shared<const DerivativeWeightedIntegrator>(
                            collocation.dt_integrator,
                            /*time_derivative_weight=*/static_cast<Real>(0.0),
                            /*non_time_derivative_weight=*/static_cast<Real>(1.0),
                            /*dt1_term_weight=*/static_cast<Real>(0.0),
                            /*dt2_term_weight=*/static_cast<Real>(0.0));
                        systems::TransientSystem transient_nd(sys, non_dt);

                        systems::SystemStateView state;
                        state.time = stage_time;
                        state.dt = dt_step;
                        state.dt_prev = history.dtPrev();
                        state.u = std::span<const Real>(Ui.data(), Ui.size());

                        auto J_view = collocation.jacobian->createAssemblyView();
                        FE_CHECK_NOT_NULL(J_view.get(), "TimeLoop: collocation jacobian view");
                        OffsetSystemView J_block(*J_view, row_offset, row_offset);
                        sys.beginTimeStep();
                        (void)transient_nd.assemble(req_matrix, state, &J_block, nullptr);
                    }
                }
            }

            rep.residual_norm = collocation.residual->norm();
            if (it == 0) {
                rep.residual_norm0 = rep.residual_norm;
            }

            const bool abs_enabled = options_.newton.abs_tolerance > 0.0;
            const bool rel_enabled = options_.newton.rel_tolerance > 0.0;
            const bool abs_ok = abs_enabled && rep.residual_norm <= options_.newton.abs_tolerance;
            const bool rel_ok = rel_enabled
                && (rep.residual_norm0 > 0.0
                       ? (rep.residual_norm / rep.residual_norm0 <= options_.newton.rel_tolerance)
                       : abs_ok);
            if (abs_ok || rel_ok) {
                rep.converged = true;
                rep.iterations = it;
                break;
            }

            // Legacy-compatible stagnation detection: once the residual has
            // decreased from its initial value, accept the best achievable
            // precision even if further reduction stalls.
            if (it > 0 && options_.newton.stagnation_tolerance > 0.0 &&
                prev_residual_norm > 0.0 && std::isfinite(prev_residual_norm) &&
                rep.residual_norm0 > 0.0 && rep.residual_norm < rep.residual_norm0) {
                const double ratio = rep.residual_norm / prev_residual_norm;
                if (ratio >= options_.newton.stagnation_tolerance) {
                    rep.converged = true;
                    rep.iterations = it;
                    break;
                }
            }
            prev_residual_norm = rep.residual_norm;

            collocation.delta->zero();
            rep.linear = linear.solve(*collocation.jacobian, *collocation.delta, *collocation.residual);
            FE_THROW_IF(!rep.linear.converged, FEException,
                        "TimeLoop: linear solve did not converge: " + rep.linear.message);

            // Newton update on stage values: U <- U - dU
            auto dU = collocation.delta->localSpan();
            FE_CHECK_ARG(dU.size() == U_all.size(), "TimeLoop: collocation delta size mismatch");
            for (std::size_t k = 0; k < U_all.size(); ++k) {
                U_all[k] -= dU[k];
            }
            if (!constraints.empty()) {
                for (int s = 0; s < method.stages; ++s) {
                    auto Ui = U_all.subspan(static_cast<std::size_t>(s) * static_cast<std::size_t>(n_dofs),
                                            static_cast<std::size_t>(n_dofs));
                    constraints.distribute(reinterpret_cast<double*>(Ui.data()),
                                           static_cast<GlobalIndex>(Ui.size()));
                }
            }

            if (options_.newton.step_tolerance > 0.0) {
                const double step_norm = collocation.delta->norm();
                if (step_norm <= options_.newton.step_tolerance) {
                    rep.converged = true;
                    rep.iterations = it + 1;
                    break;
                }
            }
        }

        if (!rep.converged) {
            rep.iterations = options_.newton.max_iterations;
            return rep;
        }
        }

        // Write u_{n+1} to history.u so TimeHistory::acceptStep shifts correctly.
        if (temporal_order == 1) {
            if (method.stiffly_accurate) {
                const int s = method.final_stage;
                FE_THROW_IF(s < 0 || s >= method.stages, InvalidArgumentException,
                            "TimeLoop: invalid stiffly-accurate final stage index");
                const auto U_final = U_all.subspan(static_cast<std::size_t>(s) * static_cast<std::size_t>(n_dofs),
                                                   static_cast<std::size_t>(n_dofs));
                auto u_out = history.uSpan();
                FE_CHECK_ARG(u_out.size() == U_final.size(), "TimeLoop: collocation output size mismatch");
                std::copy(U_final.begin(), U_final.end(), u_out.begin());
            } else {
                auto u_out = history.uSpan();
                FE_CHECK_ARG(u_out.size() == u_n.size(), "TimeLoop: collocation output size mismatch");
                std::copy(u_n.begin(), u_n.end(), u_out.begin());
                for (int j = 0; j < method.stages; ++j) {
                    const double wj = method.final_w[static_cast<std::size_t>(j)];
                    const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                  static_cast<std::size_t>(n_dofs));
                    for (std::size_t k = 0; k < u_out.size(); ++k) {
                        u_out[k] += static_cast<Real>(wj) * (Uj[k] - u_n[k]);
                    }
                }
            }
        } else {
            FE_CHECK_NOT_NULL(so_data, "TimeLoop: collocation second-order data");

            auto u_out = history.uSpan();
            if (method.stiffly_accurate) {
                const int s = method.final_stage;
                FE_THROW_IF(s < 0 || s >= method.stages, InvalidArgumentException,
                            "TimeLoop: invalid stiffly-accurate final stage index");
                const auto U_final = U_all.subspan(static_cast<std::size_t>(s) * static_cast<std::size_t>(n_dofs),
                                                   static_cast<std::size_t>(n_dofs));
                FE_CHECK_ARG(u_out.size() == U_final.size(), "TimeLoop: collocation output size mismatch");
                std::copy(U_final.begin(), U_final.end(), u_out.begin());
            } else {
                const auto dv0 = collocation.dv0->localSpan();
                FE_CHECK_ARG(u_out.size() == u_n.size(), "TimeLoop: collocation output size mismatch");
                for (std::size_t k = 0; k < u_out.size(); ++k) {
                    u_out[k] = static_cast<Real>(so_data->u1_u0) * u_n[k] +
                        static_cast<Real>(so_data->u1_dv0) * dv0[k];
                }
                for (int j = 0; j < method.stages; ++j) {
                    const double wj = so_data->u1[static_cast<std::size_t>(j)];
                    const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                  static_cast<std::size_t>(n_dofs));
                    for (std::size_t k = 0; k < u_out.size(); ++k) {
                        u_out[k] += static_cast<Real>(wj) * Uj[k];
                    }
                }
            }
        }

        if (!constraints.empty()) {
            auto u_out = history.uSpan();
            constraints.distribute(reinterpret_cast<double*>(u_out.data()),
                                   static_cast<GlobalIndex>(u_out.size()));
        }

        return rep;
    };

    const VSVO_BDF_Controller* vsvo_controller = nullptr;
    std::vector<std::shared_ptr<const systems::TimeIntegrator>> vsvo_integrators;
    std::unique_ptr<backends::GenericVector> vsvo_pred{};
    int order_next = 0;

    const double nominal_time_span = std::max({
        1.0,
        std::abs(t_end),
        std::abs(options_.t0),
        std::abs(t_end - options_.t0),
        static_cast<double>(options_.max_steps) * std::abs(options_.dt),
    });
    const double time_tol =
        1000.0 * std::numeric_limits<double>::epsilon() * nominal_time_span;

    const bool adaptive = static_cast<bool>(options_.step_controller);
    const int max_retries = adaptive ? std::max(0, options_.step_controller->maxRetries()) : 0;
    double dt_next = options_.dt;
    // Reused by every attempt. Once rate state exists, this owns at most one
    // checkpoint vector per rate, so adaptive runs pay O(n_dofs) copies but no
    // per-step backend-vector allocation.
    TimeHistory::RateStateSnapshot attempt_rate_state_snapshot;

    auto adjustStepToFinalInterval = [&](double time, double candidate_dt) {
        double dt_adjusted = candidate_dt;
        if (!options_.adjust_last_step) {
            return dt_adjusted;
        }

        const double remaining = t_end - time;
        if (remaining < dt_adjusted) {
            return remaining;
        }

        const double fraction = options_.last_step_absorb_fraction;
        if (fraction > 0.0 && dt_adjusted > 0.0) {
            const double terminal_remainder = remaining - dt_adjusted;
            if (terminal_remainder > 0.0 &&
                terminal_remainder <= fraction * dt_adjusted) {
                return remaining;
            }
        }

        return dt_adjusted;
    };

	    if (options_.scheme == SchemeKind::VSVO_BDF) {
	        FE_THROW_IF(!adaptive, InvalidArgumentException,
	                    "TimeLoop: VSVO_BDF scheme requires a step_controller");
	        vsvo_controller = dynamic_cast<const VSVO_BDF_Controller*>(options_.step_controller.get());
	        FE_THROW_IF(vsvo_controller == nullptr, InvalidArgumentException,
	                    "TimeLoop: VSVO_BDF scheme requires a VSVO_BDF_Controller");
	        const int system_temporal_order = transient.system().temporalOrder();
	        FE_THROW_IF(system_temporal_order > 2, NotImplementedException,
	                    "TimeLoop: VSVO_BDF supports temporal order <= 2");
	        const int deriv_order = (system_temporal_order >= 2) ? 2 : 1;
	        FE_THROW_IF(history.historyDepth() < vsvo_controller->maxOrder() + deriv_order, InvalidArgumentException,
	                    "TimeLoop: VSVO_BDF requires history depth >= max_order + temporal_order");

	        const int max_order = vsvo_controller->maxOrder();
	        vsvo_integrators.resize(static_cast<std::size_t>(max_order + 1));
	        for (int p = 1; p <= max_order; ++p) {
	            vsvo_integrators[static_cast<std::size_t>(p)] = std::make_shared<const systems::BDFIntegrator>(p);
        }

	        vsvo_pred = factory.createVector(n_dofs);
	        FE_CHECK_NOT_NULL(vsvo_pred.get(), "TimeLoop: vsvo_pred");
	        order_next = vsvo_controller->initialOrder();
	    }

    for (int step = 0; step < options_.max_steps; ++step) {
        const double t = history.time();
        if (t + time_tol >= t_end) {
            report.success = true;
            report.steps_taken = step;
            report.final_time = t_end;
            history.setTime(t_end);
            return report;
        }

        double dt = dt_next;
        FE_THROW_IF(!(dt > 0.0) || !std::isfinite(dt), systems::InvalidStateException, "TimeLoop: invalid dt");
        int order = order_next;

        const double remaining0 = t_end - t;
        if (options_.adjust_last_step) {
            if (remaining0 <= time_tol) {
                report.success = true;
                report.steps_taken = step;
                report.final_time = t_end;
                history.setTime(t_end);
                return report;
            }
            dt = adjustStepToFinalInterval(t, dt);
        }

        bool accepted = false;
        NewtonReport nr;

        for (int attempt = 0; attempt <= max_retries; ++attempt) {
            // Candidate-rate updates are transactional until this attempt
            // reaches the irreversible system/history acceptance boundary.
            // Every earlier continue/return/throw path restores both values
            // and allocation state automatically.
            AttemptRateStateGuard attempt_rate_state(
                history, factory, attempt_rate_state_snapshot);

            const double remaining = t_end - t;
            if (options_.adjust_last_step) {
                if (remaining <= time_tol) {
                    accepted = true;
                    report.success = true;
                    report.steps_taken = step;
                    report.final_time = t_end;
                    history.setTime(t_end);
                    return report;
                }
                dt = adjustStepToFinalInterval(t, dt);
            }

            history.setDt(dt);
            history.resetCurrentToPrevious();
            if (callbacks.on_step_start) {
                callbacks.on_step_start(history);
            }

            const double dt_prev_step = history.dtPrev();

            const double solve_time = t + dt;
            if (callbacks.on_before_physics_solve) {
                const bool moving_domain_ok = callbacks.on_before_physics_solve(history, solve_time, dt);
                FE_THROW_IF(!moving_domain_ok, FEException,
                            "TimeLoop: before-physics-solve callback rejected the step");
            }
            ensure_workspace_matches_sparsity();
            transient.system().beginTimeStep();

            int scheme_order = 0;
            double error_norm = -1.0;
            double error_norm_low = -1.0;
            double error_norm_high = -1.0;
            bool used_collocation = false;
            CollocationFamily collocation_family_used = CollocationFamily::Gauss;
            std::optional<double> monolithic_aux_stage_alpha_f{};
            int collocation_stages_used = 0;
            bool generalized_alpha_first_order_rate_n_saved = false;

            bool threw = false;
            std::exception_ptr caught_exception{};
            try {
                if (options_.scheme == SchemeKind::BackwardEuler || options_.scheme == SchemeKind::DG0) {
                    nr = newton.solveStep(transient, linear, solve_time, history, workspace);
                } else if (options_.scheme == SchemeKind::BDF2) {
                    if (history.stepIndex() < 1) {
                        // Use a 2nd-order starter (Crank–Nicolson) so the global BDF2
                        // scheme reaches its expected temporal order.
                        nr = solveThetaStep(/*theta=*/0.5, solve_time, dt);
                    } else {
                        systems::TransientSystem transient_step(transient.system(), bdf2);
                        nr = newton.solveStep(transient_step, linear, solve_time, history, workspace);
                    }
                } else if (options_.scheme == SchemeKind::ThetaMethod) {
                    nr = solveThetaStep(options_.theta, solve_time, dt);
                } else if (options_.scheme == SchemeKind::Newmark) {
                    const int temporal_order = transient.system().temporalOrder();
                    if (temporal_order <= 1) {
                        nr = solveThetaStep(/*theta=*/0.5, solve_time, dt);
                    } else if (temporal_order == 2) {
                        FE_CHECK_NOT_NULL(newmark_beta.get(), "TimeLoop: NewmarkBeta integrator");
                        const bool had_u_dot = history.hasUDotState();
                        const bool had_u_ddot = history.hasUDDotState();
                        ensureSecondOrderKinematics(/*overwrite_u_dot=*/!had_u_dot,
                                                    /*overwrite_u_ddot=*/!had_u_ddot,
                                                    /*require_u_ddot=*/!had_u_ddot);

                        // Ensure (u_n, v_n, a_n) values are ghost-consistent before constructing constants.
                        updateGhostsAndDistributeHistory(transient.system().constraints(), history);

                        // Save displacement history since we overwrite the first two slots with
                        // scheme-specific constant vectors.
                        copyVector(*scratch_vec1, history.uPrev());
                        copyVector(*scratch_vec2, history.uPrev2());

                        struct RestoreGuard {
                            TimeHistory& history;
                            backends::GenericVector& saved_prev;
                            backends::GenericVector& saved_prev2;
                            ~RestoreGuard()
                            {
                                copyVector(history.uPrev(), saved_prev);
                                copyVector(history.uPrev2(), saved_prev2);
                            }
                        } restore{history, *scratch_vec1, *scratch_vec2};

                        const auto u_n = scratch_vec1->localSpan();
                        const auto v_n = history.uDotSpan();
                        const auto a_n = history.uDDotSpan();

                        writeStructuralHistoryConstants(/*alpha_m=*/1.0,
                                                       /*alpha_f=*/1.0,
                                                       /*beta=*/options_.newmark_beta,
                                                       /*gamma=*/options_.newmark_gamma,
                                                       dt,
                                                       u_n,
                                                       v_n,
                                                       a_n,
                                                       history.uPrev().localSpan(),
                                                       history.uPrev2().localSpan());

                        systems::TransientSystem transient_step(transient.system(), newmark_beta);
                        nr = newton.solveStep(transient_step, linear, solve_time, history, workspace);
                    } else {
                        FE_THROW(NotImplementedException, "TimeLoop: Newmark supports temporal order <= 2");
                    }
                } else if (options_.scheme == SchemeKind::DG1 || options_.scheme == SchemeKind::DG ||
                           options_.scheme == SchemeKind::CG2 || options_.scheme == SchemeKind::CG) {
                    CollocationFamily family = CollocationFamily::Gauss;
                    int degree = 1;
                    if (options_.scheme == SchemeKind::DG1) {
                        family = CollocationFamily::RadauIIA;
                        degree = 1;
                    } else if (options_.scheme == SchemeKind::DG) {
                        family = CollocationFamily::RadauIIA;
                        degree = (order > 0) ? order : options_.dg_degree;
                        degree = std::max(0, std::min(10, degree));
                    } else if (options_.scheme == SchemeKind::CG2) {
                        family = CollocationFamily::Gauss;
                        degree = 2;
                    } else {
                        family = CollocationFamily::Gauss;
                        degree = (order > 0) ? order : options_.cg_degree;
                        degree = std::max(1, std::min(10, degree));
                    }

                    if (family == CollocationFamily::RadauIIA && degree == 0) {
                        // dG(0) is Backward Euler.
                        scheme_order = 1;
                        nr = newton.solveStep(transient, linear, solve_time, history, workspace);
                    } else {
                        const int stages = (family == CollocationFamily::RadauIIA) ? (degree + 1) : degree;
                        const auto& method = getCollocationMethod(family, stages);
                        used_collocation = true;
                        collocation_family_used = family;
                        collocation_stages_used = method.stages;
                        scheme_order = method.order;
                        nr = solveCollocationStep(method, t, dt);
                    }
                } else if (options_.scheme == SchemeKind::CG1) {
                    nr = solveThetaStep(/*theta=*/0.5, solve_time, dt);
                } else if (options_.scheme == SchemeKind::GeneralizedAlpha) {
                    const int temporal_order = transient.system().temporalOrder();
	                    if (temporal_order <= 1) {
	                        FE_CHECK_NOT_NULL(generalized_alpha_fo.get(), "TimeLoop: generalized-alpha(1st-order) integrator");
	                        FE_THROW_IF(!ga1_params.has_value(), systems::InvalidStateException,
	                                    "TimeLoop: generalized-alpha parameters not initialized");

	                        auto& sys = transient.system();
	                        const auto& constraints = sys.constraints();
                            const auto dt_fields = sys.timeDerivativeFields(options_.newton.jacobian_op);
                            const auto nondt_dofs = collectNonTimeDerivativeDofs(sys, dt_fields);

	                        // Ensure uDot storage exists and is initialized before the stage solve.
	                        const bool had_u_dot = history.hasUDotState();
	                        history.ensureSecondOrderState(factory);
                        if (!had_u_dot) {
                            // Compute an optional linearized initial-rate estimate by separating compiled
                            // terms tagged with a time derivative from terms without one. For a separable
                            // first-order residual M*dt(u) + F(u), this is the usual consistent-rate mass
                            // solve. A nonlinear term that nests dt(u) with spatial state remains one tagged
                            // term, so this construction is deliberately an estimate rather than an exact
                            // nonlinear DAE consistency solve. It still avoids the degenerate uDot=0 startup
                            // for supported transient formulations.
                            history.uDot().zero();

                            if (!options_.initialize_first_order_rate_from_pde) {
                                (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                                    history,
                                    history.uDot().localSpan(),
                                    history.uDDot().localSpan(),
                                    /*overwrite_u_dot=*/true,
                                    /*overwrite_u_ddot=*/false);
                            } else if (!dt_fields.empty()) {
                                // Initial consistency is an equation for
                                // uDot_n at the accepted state u_n. Explicit
                                // forcing and time-dependent constraints must
                                // therefore be evaluated at t_n, not at the
                                // subsequent generalized-alpha stage time.
                                const double initialization_time = t;
                                sys.updateConstraints(initialization_time, dt);
                                ensure_workspace_matches_sparsity();
                                sys.beginTimeStep();
                                const auto init_dirichlet_dofs = setLinearDirichletDofs(linear, constraints);

                                systems::SystemStateView init_state{};
                                init_state.time = initialization_time;
                                init_state.dt = dt;
                                init_state.dt_prev = history.dtPrev();
                                init_state.u = history.uSpan();
                                init_state.u_prev = history.uPrevSpan();
                                // GeneralizedAlphaFirstOrderIntegrator with
                                // history_rate_order==0 reserves history slot 2
                                // for the injected rate uDot^n.  The regular
                                // displacement history must not be exposed in
                                // that slot while constructing or assembling
                                // the initialization operator.
                                init_state.u_prev2 = history.uDotSpan();
                                init_state.u_vector = &history.u();
                                init_state.u_prev_vector = &history.uPrev();
                                init_state.u_prev2_vector = &history.uDot();
                                std::vector<std::span<const Real>> init_u_history(
                                    history.uHistorySpans().begin(),
                                    history.uHistorySpans().end());
                                FE_THROW_IF(init_u_history.size() < 2u,
                                            systems::InvalidStateException,
                                            "TimeLoop: first-order generalized-alpha rate initialization requires two history slots");
                                init_u_history[1] = history.uDotSpan();
                                init_state.u_history = init_u_history;
                                init_state.dt_history = history.dtHistory();

                                auto ctx_base = generalized_alpha_fo->buildContext(temporal_order, init_state);
                                const auto* dt1 = ctx_base.dt1 ? &(*ctx_base.dt1) : nullptr;
                                const double c = dt1 ? static_cast<double>(dt1->coeff(/*history_index=*/0)) : 0.0;

                                const bool can_solve = (dt1 != nullptr) && std::isfinite(c) && (std::abs(c) > 0.0);

                                if (can_solve) {
                                    // Assemble the residual from compiled terms that contain no dt terminal.
                                    assembly::TimeIntegrationContext ctx_non_dt = ctx_base;
                                    ctx_non_dt.time_derivative_term_weight = 0.0;
                                    ctx_non_dt.non_time_derivative_term_weight = 1.0;

                                    auto& r_non_dt = *scratch_vec0;
                                    r_non_dt.zero();
                                    auto r_view = r_non_dt.createAssemblyView();
                                    FE_CHECK_NOT_NULL(r_view.get(), "TimeLoop: uDot init residual view");

                                    systems::AssemblyRequest req_r;
                                    req_r.op = options_.newton.residual_op;
                                    req_r.want_vector = true;
                                    systems::SystemStateView state_r = init_state;
                                    state_r.time_integration = &ctx_non_dt;
                                    const auto ar_r = sys.assemble(req_r, state_r, nullptr, r_view.get());
                                    FE_THROW_IF(!ar_r.success, FEException,
                                                "TimeLoop: uDot initialization residual assembly failed: " + ar_r.error_message);

                                    // Assemble the Jacobian of compiled terms tagged with a dt terminal, with
                                    // all untagged terms disabled.
                                    assembly::TimeIntegrationContext ctx_dt_only = ctx_base;
                                    ctx_dt_only.time_derivative_term_weight = 1.0;
                                    ctx_dt_only.non_time_derivative_term_weight = 0.0;

                                    auto& A = *workspace.jacobian;
                                    A.zero();
                                    auto A_view = A.createAssemblyView();
                                    FE_CHECK_NOT_NULL(A_view.get(), "TimeLoop: uDot init matrix view");

                                    systems::AssemblyRequest req_A;
                                    req_A.op = options_.newton.jacobian_op;
                                    req_A.want_matrix = true;
                                    systems::SystemStateView state_A = init_state;
                                    state_A.time_integration = &ctx_dt_only;
                                    const auto ar_A = sys.assemble(req_A, state_A, A_view.get(), nullptr);
                                    FE_THROW_IF(!ar_A.success, FEException,
                                                "TimeLoop: uDot initialization matrix assembly failed: " + ar_A.error_message);

                                    // Build RHS b = -c * r_non_dt. For a separable residual this gives
                                    //   (c*M) * uDot = -c * F(u)  =>  M * uDot = -F(u).
                                    // Mixed nonlinear dt terms instead produce the documented linearized
                                    // startup estimate.
                                    auto& b = *scratch_vec1;
                                    copyVector(b, r_non_dt);
                                    b.scale(static_cast<Real>(-c));
                                    if (!nondt_dofs.empty()) {
                                        auto b_mod = b.createAssemblyView();
                                        FE_CHECK_NOT_NULL(b_mod.get(), "TimeLoop: uDot init rhs view");
                                        b_mod->beginAssemblyPhase();
                                        b_mod->zeroVectorEntries(nondt_dofs);
                                        b_mod->finalizeAssembly();

                                        auto A_mod = A.createAssemblyView();
                                        FE_CHECK_NOT_NULL(A_mod.get(), "TimeLoop: uDot init matrix modify view");
                                        A_mod->beginAssemblyPhase();
                                        A_mod->zeroRows(nondt_dofs, /*set_diagonal=*/true);
                                        A_mod->finalizeAssembly();
                                    }

                                    // Cut-domain differential fields are
                                    // field-wide in the symbolic dt scan, but
                                    // their mass operator is present only on
                                    // the retained physical side.  Rows on
                                    // the complementary side are exact zeros
                                    // in this dt-only initialization problem.
                                    // They represent no rate equation; make
                                    // those rates homogeneous instead of
                                    // passing a singular matrix to the linear
                                    // backend.  The subsequent monolithic
                                    // stage still solves their algebraic
                                    // extension equations normally.
                                    const auto exact_zero_mass_dofs =
                                        collectOwnedExactZeroMassRows(
                                            sys,
                                            A,
                                            options_.newton.jacobian_op);
                                    if (!exact_zero_mass_dofs.empty()) {
                                        auto b_zero = b.createAssemblyView();
                                        FE_CHECK_NOT_NULL(
                                            b_zero.get(),
                                            "TimeLoop: uDot zero-mass rhs view");
                                        b_zero->beginAssemblyPhase();
                                        b_zero->zeroVectorEntries(
                                            exact_zero_mass_dofs);
                                        b_zero->finalizeAssembly();

                                        auto A_zero = A.createAssemblyView();
                                        FE_CHECK_NOT_NULL(
                                            A_zero.get(),
                                            "TimeLoop: uDot zero-mass matrix view");
                                        A_zero->beginAssemblyPhase();
                                        A_zero->zeroRows(
                                            exact_zero_mass_dofs,
                                            /*set_diagonal=*/true);
                                        A_zero->finalizeAssembly();
                                    }

                                    // Solve A * uDot = b.
                                    // The dt-only Jacobian assembled above can be structurally incompatible with
                                    // certain specialized saddle-point solvers (e.g., block-Schur), since it
                                    // intentionally disables all non-dt terms and may eliminate required coupling
                                    // blocks. For this one-time initialization solve, fall back to a generic Krylov
                                    // method when the configured linear solver is block-Schur.
                                    const auto saved_opts = linear.getOptions();
                                    struct RestoreSolverOptionsGuard {
                                        backends::LinearSolver& linear;
                                        backends::SolverOptions opts;
                                        ~RestoreSolverOptionsGuard() noexcept
                                        {
                                            try {
                                                linear.setOptions(opts);
                                            } catch (...) {
                                            }
                                        }
                                    } restore_linear_opts{linear, saved_opts};

                                    backends::SolverOptions init_opts = saved_opts;
                                    if (init_opts.method == backends::SolverMethod::BlockSchur) {
                                        init_opts.method = backends::SolverMethod::GMRES;
                                        init_opts.max_iter = std::max(init_opts.max_iter, 50);
                                        linear.setOptions(init_opts);
                                    }

                                    backends::SolverReport rep{};
                                    try {
                                        logInitializationSolveDiagnostics(
                                            "u_dot", constraints, init_dirichlet_dofs, A, b);
                                        rep = linear.solve(A, history.uDot(), b);
                                    } catch (const std::exception&) {
                                        rep.converged = false;
                                        rep.message = "linear_solve_exception";
                                    }

                                    const Real solved_rate_norm = history.uDot().norm();
                                    bool accept_udot_init_solve =
                                        rep.converged &&
                                        std::isfinite(
                                            static_cast<double>(solved_rate_norm));
                                    if (!accept_udot_init_solve &&
                                        std::isfinite(rep.initial_residual_norm) &&
                                        std::isfinite(rep.final_residual_norm)) {
                                        const Real rhs_norm = std::max<Real>(
                                            static_cast<Real>(rep.initial_residual_norm),
                                            static_cast<Real>(1e-30));
                                        const Real requested_target = std::max(
                                            init_opts.abs_tol,
                                            init_opts.rel_tol * rhs_norm);
                                        const bool meets_requested_target =
                                            std::isfinite(static_cast<double>(requested_target)) &&
                                            rep.final_residual_norm <= requested_target;

                                        bool meets_nonlinear_floor = false;
                                        if (options_.newton.abs_tolerance > 0.0 &&
                                            std::isfinite(rep.relative_residual)) {
                                            const Real nonlinear_floor =
                                                static_cast<Real>(options_.newton.abs_tolerance);
                                            const Real relaxed_relative_target =
                                                init_opts.rel_tol > 0.0
                                                    ? static_cast<Real>(2.0) * init_opts.rel_tol
                                                    : std::numeric_limits<Real>::infinity();
                                            meets_nonlinear_floor =
                                                std::isfinite(static_cast<double>(nonlinear_floor)) &&
                                                std::isfinite(static_cast<double>(relaxed_relative_target)) &&
                                                rep.final_residual_norm <= nonlinear_floor &&
                                                rep.relative_residual <= relaxed_relative_target;
                                        }

                                        accept_udot_init_solve =
                                            std::isfinite(static_cast<double>(
                                                solved_rate_norm)) &&
                                            (meets_requested_target ||
                                             meets_nonlinear_floor);
                                    }

                                    if (!accept_udot_init_solve) {
                                        // Fall back to a finite-difference uDot (may be zero at the first step).
                                        (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                                            history,
                                            history.uDot().localSpan(),
                                            history.uDDot().localSpan(),
                                            /*overwrite_u_dot=*/true,
                                            /*overwrite_u_ddot=*/false);
                                    } else {
                                        rep.converged = true;
                                    }
                                    logFirstOrderRateInitializationResult(
                                        rep,
                                        accept_udot_init_solve,
                                        exact_zero_mass_dofs.size(),
                                        history.uDot().norm(),
                                        accept_udot_init_solve
                                            ? "none"
                                            : (rep.message.empty()
                                                   ? "linear_solve_not_accepted"
                                                   : rep.message));
                                } else {
                                    (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                                        history,
                                        history.uDot().localSpan(),
                                        history.uDDot().localSpan(),
                                        /*overwrite_u_dot=*/true,
                                        /*overwrite_u_ddot=*/false);
                                }
                            }
                        }
	                        if (!nondt_dofs.empty()) {
	                            zeroVectorEntries(nondt_dofs, history.uDot());
	                        }
	                        // Keep the constraint-consistent rate history at strong-Dirichlet
	                        // DOFs on stage entry. Zeroing here hands the stage solve an
	                        // injected wall rate of 0 instead of the discrete g_dot carried by
	                        // the end-of-step update, so the alpha_m-weighted mass term sees a
	                        // spurious wall acceleration and pumps a y-alternating velocity
	                        // error into the first interior cell row for moving-wall data
	                        // (open-vessel MMS sawtooth; constant Dirichlet data is unaffected
	                        // because its finite-difference rate is already zero). True MPC
	                        // slave rates stay consistent without redistribution because the
	                        // rate update is linear in the values.
	                        // The one exception is first-rate initialization: the
	                        // linearized uDot solve writes Dirichlet-row artifacts
	                        // (~ -c * r at constrained rows, growing as 1/dt) into the
	                        // constrained entries, which must be sanitized before the first
	                        // stage solve or Newton stalls (seen on MMS nx16 dt=0.01). Set
	                        // SVMP_ZERO_CONSTRAINED_RATES=1 to restore zeroing on every step.
	                        if (!constraints.empty() &&
	                            (!had_u_dot || zeroConstrainedRatesRequested())) {
	                            constraints.distributeHomogeneous(history.uDot());
	                        }

                        // Ensure (u_n, uDot_n) values are ghost-consistent before constructing constants.
                        updateGhostsAndDistributeHistory(transient.system().constraints(), history);

                        // Save displacement history (u^{n-1}) since we overwrite uPrev2 with uDot^n
                        // for the stage solve.
                        copyVector(*scratch_vec2, history.uPrev2());

                        struct RestoreGuard {
                            TimeHistory& history;
                            backends::GenericVector& saved_prev2;
                            ~RestoreGuard()
                            {
                                copyVector(history.uPrev2(), saved_prev2);
                            }
                        } restore{history, *scratch_vec2};

                        // Inject uDot^n into the u^{n-1} history slot; the integrator uses it via
                        // history_rate_order==0 to keep generalized-α one-step in (u,uDot).
                        copyVector(history.uPrev2(), history.uDot());

                        systems::TransientSystem transient_stage(transient.system(), generalized_alpha_fo);
                        const double stage_time = t + ga1_params->alpha_f * dt;

                        // Predictor for the stage unknown u_{n+alpha_f}: start from u^n and extrapolate using uDot^n.
                        {
                            auto cur = history.uSpan();
                            const auto v = history.uDotSpan();
                            FE_CHECK_ARG(cur.size() == v.size(), "TimeLoop: generalized-alpha predictor size mismatch");
                            const Real alpha_dt = static_cast<Real>(ga1_params->alpha_f * dt);
                            for (std::size_t i = 0; i < cur.size(); ++i) {
                                cur[i] += alpha_dt * v[i];
                            }
                            constraints.updateGhostsAndDistribute(history.u());
                        }

	                        nr = newton.solveStep(transient_stage, linear, stage_time, history, workspace);
                        if (nr.converged) {
                            const double inv_af = 1.0 / ga1_params->alpha_f;
                            const double c_prev = (ga1_params->alpha_f - 1.0) * inv_af;
	                            if (!nondt_dofs.empty()) {
	                                copyVector(*scratch_vec0, history.u());
	                            }
	                            auto cur = history.uSpan();
	                            const auto prev = history.uPrevSpan();
	                            FE_CHECK_ARG(cur.size() == prev.size(), "TimeLoop: generalized-alpha size mismatch");
	                            for (std::size_t i = 0; i < cur.size(); ++i) {
	                                cur[i] = static_cast<Real>(inv_af) * cur[i] + static_cast<Real>(c_prev) * prev[i];
	                            }
	                            if (!nondt_dofs.empty()) {
	                                // Algebraic fields (e.g., pressure for incompressible flow) should not be
	                                // extrapolated via the generalized-α stage relation. Preserve the stage
	                                // values returned by the nonlinear solve for these DOFs.
	                                copyVectorEntries(nondt_dofs, history.u(), *scratch_vec0);
	                            }

	                            // Update uDot_{n+1} (stored as TimeHistory::uDot) for use by later stages
	                            // and end-of-step finalization.
	                            const double gamma = ga1_params->gamma;
                            const double inv_gamma_dt = 1.0 / (gamma * dt);
                            const double c_old = (1.0 - gamma) / gamma;
                            auto v = history.uDotSpan();
                            FE_CHECK_ARG(v.size() == cur.size(), "TimeLoop: generalized-alpha uDot size mismatch");
	                            copyVector(*generalized_alpha_rate_n,
	                                       history.uDot());
	                            generalized_alpha_first_order_rate_n_saved =
	                                true;
	                            for (std::size_t i = 0; i < cur.size(); ++i) {
	                                const Real v_n = v[i];
	                                v[i] = static_cast<Real>(inv_gamma_dt) * (cur[i] - prev[i]) -
	                                    static_cast<Real>(c_old) * v_n;
	                            }
	                            if (!nondt_dofs.empty()) {
	                                zeroVectorEntries(nondt_dofs, history.uDot());
	                            }
                            // Do NOT distribute the homogeneous constraints into uDot here.
                            // The update above already produces constraint-consistent rates:
                            // constrained VALUES satisfy the (possibly time-dependent) affine
                            // relations at both time levels, so the finite-difference rate
                            // carries the inhomogeneity rate g_dot(t) automatically.
                            // distributeHomogeneous would zero g_dot at Dirichlet DOFs, so the
                            // injected-rate stencil sees a wall acceleration of (alpha_m/gamma)
                            // times the true value; the consistent-mass coupling then pumps a
                            // secular, h- and dt-independent velocity error into wall-adjacent
                            // cells for moving-wall data (observed as the bottom-wall sawtooth
                            // in the open-vessel MMS). Set SVMP_ZERO_CONSTRAINED_RATES=1 to
                            // restore the legacy zeroing for comparison.
                            if (!constraints.empty() && zeroConstrainedRatesRequested()) {
                                constraints.distributeHomogeneous(history.uDot());
                            }
                            monolithic_aux_stage_alpha_f = ga1_params->alpha_f;
                        }
                    } else if (temporal_order == 2) {
                        if (!ga2_params.has_value()) {
                            ga2_params = utils::generalizedAlphaSecondOrderFromRhoInf(options_.generalized_alpha_rho_inf);
                            generalized_alpha_so = std::make_shared<const GeneralizedAlphaSecondOrderIntegrator>(
                                GeneralizedAlphaSecondOrderIntegratorOptions{
                                    .alpha_m = ga2_params->alpha_m,
                                    .alpha_f = ga2_params->alpha_f,
                                    .beta = ga2_params->beta,
                                    .gamma = ga2_params->gamma,
                                });
                        }
                        FE_CHECK_NOT_NULL(generalized_alpha_so.get(), "TimeLoop: generalized-alpha(2nd-order) integrator");
                        const bool had_u_dot = history.hasUDotState();
                        const bool had_u_ddot = history.hasUDDotState();
                        ensureSecondOrderKinematics(/*overwrite_u_dot=*/!had_u_dot,
                                                    /*overwrite_u_ddot=*/!had_u_ddot,
                                                    /*require_u_ddot=*/!had_u_ddot);

                        // Ensure (u_n, v_n, a_n) values are ghost-consistent before constructing constants.
                        updateGhostsAndDistributeHistory(transient.system().constraints(), history);

                        // Save displacement history since we overwrite the first two slots with
                        // scheme-specific constant vectors.
                        copyVector(*scratch_vec1, history.uPrev());
                        copyVector(*scratch_vec2, history.uPrev2());

                        struct RestoreGuard {
                            TimeHistory& history;
                            backends::GenericVector& saved_prev;
                            backends::GenericVector& saved_prev2;
                            ~RestoreGuard()
                            {
                                copyVector(history.uPrev(), saved_prev);
                                copyVector(history.uPrev2(), saved_prev2);
                            }
                        } restore{history, *scratch_vec1, *scratch_vec2};

                        const auto u_n = scratch_vec1->localSpan();
                        const auto v_n = history.uDotSpan();
                        const auto a_n = history.uDDotSpan();

                        writeStructuralHistoryConstants(ga2_params->alpha_m,
                                                       ga2_params->alpha_f,
                                                       ga2_params->beta,
                                                       ga2_params->gamma,
                                                       dt,
                                                       u_n,
                                                       v_n,
                                                       a_n,
                                                       history.uPrev().localSpan(),
                                                       history.uPrev2().localSpan());

                        systems::TransientSystem transient_stage(transient.system(), generalized_alpha_so);
                        const double stage_time = t + ga2_params->alpha_f * dt;
                        nr = newton.solveStep(transient_stage, linear, stage_time, history, workspace);
                        if (nr.converged) {
                            const double inv_af = 1.0 / ga2_params->alpha_f;
                            const double c_prev = (ga2_params->alpha_f - 1.0) * inv_af;
                            auto cur = history.uSpan();
                            FE_CHECK_ARG(cur.size() == u_n.size(), "TimeLoop: generalized-alpha(2nd-order) size mismatch");
                            for (std::size_t i = 0; i < cur.size(); ++i) {
                                cur[i] = static_cast<Real>(inv_af) * cur[i] + static_cast<Real>(c_prev) * u_n[i];
                            }
                            monolithic_aux_stage_alpha_f = ga2_params->alpha_f;
                        }
                    } else {
                        FE_THROW(NotImplementedException, "TimeLoop: GeneralizedAlpha supports temporal order <= 2");
                    }
	                } else if (options_.scheme == SchemeKind::VSVO_BDF) {
	                    FE_CHECK_NOT_NULL(vsvo_controller, "TimeLoop: VSVO_BDF controller");
	                    FE_CHECK_NOT_NULL(vsvo_pred.get(), "TimeLoop: VSVO_BDF predictor");
	                    const int system_temporal_order = transient.system().temporalOrder();

	                    order = std::max(vsvo_controller->minOrder(), std::min(order, vsvo_controller->maxOrder()));
	                    if (system_temporal_order == 2) {
	                        // VSVO_BDF is primarily intended for first-order systems. For dt(·,2) problems,
	                        // restrict to order 1 and rely on the embedded Newmark reference for error control.
	                        order = 1;
	                    }
	                    // Starter ramp: for LTE-based control, order p needs "real" history through u^{n-p}.
	                    const int max_order_by_history = std::max(1, history.stepIndex());
	                    order = std::max(vsvo_controller->minOrder(),
	                                     std::min(order, std::min(vsvo_controller->maxOrder(), max_order_by_history)));
	                    scheme_order = order;

	                    const bool need_startup_reference =
	                        (order == 1) &&
	                        ((system_temporal_order <= 1 && history.stepIndex() == 0) || system_temporal_order == 2);
	                    bool have_reference_solution = false;
	                    if (need_startup_reference) {
                        // Bootstrap VSVO error estimation on the very first step. With no "real" history
                        // yet, extrapolation-based predictors reduce to u_pred = u^n and the correction
                        // u^{n+1} - u_pred measures solution change (O(dt)), not the local truncation
                        // error (O(dt^2) for BDF1). Use an embedded pair (BE vs CN) to obtain an
                        // O(dt^2) error estimate without requiring additional history.
                        NewtonReport nr_ref;
	                        if (system_temporal_order <= 1) {
	                            nr_ref = solveThetaStep(/*theta=*/0.5, solve_time, dt);
	                        } else if (system_temporal_order == 2) {
	                            auto newmark_ref = std::make_shared<const NewmarkBetaIntegrator>(NewmarkBetaIntegratorOptions{
	                                .beta = options_.newmark_beta,
	                                .gamma = options_.newmark_gamma,
	                            });

                            const bool had_u_dot = history.hasUDotState();
                            const bool had_u_ddot = history.hasUDDotState();
                            ensureSecondOrderKinematics(/*overwrite_u_dot=*/!had_u_dot,
                                                        /*overwrite_u_ddot=*/!had_u_ddot,
                                                        /*require_u_ddot=*/!had_u_ddot);

                            updateGhostsAndDistributeHistory(transient.system().constraints(), history);

                            copyVector(*scratch_vec0, history.uPrev());
                            copyVector(*scratch_vec2, history.uPrev2());

                            struct RestoreGuard {
                                TimeHistory& history;
                                backends::GenericVector& saved_prev;
                                backends::GenericVector& saved_prev2;
                                ~RestoreGuard()
                                {
                                    copyVector(history.uPrev(), saved_prev);
                                    copyVector(history.uPrev2(), saved_prev2);
                                }
                            } restore{history, *scratch_vec0, *scratch_vec2};

                            const auto u_n = scratch_vec0->localSpan();
                            const auto v_n = history.uDotSpan();
                            const auto a_n = history.uDDotSpan();

                            writeStructuralHistoryConstants(/*alpha_m=*/1.0,
                                                           /*alpha_f=*/1.0,
                                                           options_.newmark_beta,
                                                           options_.newmark_gamma,
                                                           dt,
                                                           u_n,
                                                           v_n,
                                                           a_n,
                                                           history.uPrev().localSpan(),
                                                           history.uPrev2().localSpan());

                            systems::TransientSystem transient_ref(transient.system(), newmark_ref);
                            nr_ref = newton.solveStep(transient_ref, linear, solve_time, history, workspace);
	                        } else {
	                            FE_THROW(NotImplementedException, "TimeLoop: VSVO_BDF supports temporal order <= 2");
	                        }
                        if (nr_ref.converged) {
                            copyVector(*scratch_vec1, history.u());
                            have_reference_solution = true;

                            // Reset internal states and solution guess before solving the actual BDF step.
                            transient.system().beginTimeStep();
                            history.resetCurrentToPrevious();
                        } else {
                            nr = nr_ref;
                            threw = false;
                            error_norm = -1.0;
                            transient.system().beginTimeStep();
                            history.resetCurrentToPrevious();
                        }
                    }

                    if (need_startup_reference && !have_reference_solution) {
                        // Reference solve failed; treat this attempt as a nonlinear failure so the
                        // controller can reduce dt and retry.
                        // `nr` already holds the reference report.
                    } else {
                    auto powInt = [](double x, int p) -> double {
                        double out = 1.0;
                        for (int i = 0; i < p; ++i) {
                            out *= x;
                        }
                        return out;
                    };

                    auto factorial = [](int n) -> double {
                        double out = 1.0;
                        for (int i = 2; i <= n; ++i) {
                            out *= static_cast<double>(i);
                        }
                        return out;
                    };

	                    auto computeLTENorm = [&](int p, double dt_step) -> double {
	                        if (p < vsvo_controller->minOrder() || p > vsvo_controller->maxOrder()) {
	                            return -1.0;
	                        }
	                        const int deriv_order = (system_temporal_order >= 2) ? 2 : 1;
	                        const int p1 = p + deriv_order;
	                        const int required_step_index = p + deriv_order - 1;
	                        const int required_history_depth = p + deriv_order;
	                        // Need `required_history_depth` "real" past states to form dd_{p1}.
	                        if (history.stepIndex() < required_step_index) {
	                            return -1.0;
	                        }
	                        if (history.historyDepth() < required_history_depth) {
	                            return -1.0;
	                        }

                        FE_THROW_IF(!(dt_step > 0.0) || !std::isfinite(dt_step),
                                    systems::InvalidStateException,
                                    "TimeLoop: VSVO_BDF invalid dt for LTE estimate");

                        const auto dt_hist = history.dtHistory();
                        const double dt_prev = (history.dtPrev() > 0.0 && std::isfinite(history.dtPrev()))
                            ? history.dtPrev()
                            : dt_step;

                        auto historyDt = [&](int idx) -> double {
                            if (idx < 0 || idx >= static_cast<int>(dt_hist.size())) {
                                return dt_prev;
                            }
                            const double v = dt_hist[static_cast<std::size_t>(idx)];
                            if (v > 0.0 && std::isfinite(v)) {
                                return v;
                            }
                            return dt_prev;
                        };

	                        // Nodes for the derivative stencil at t_{n+1} shifted by t_{n+1}.
	                        const int method_points = p + deriv_order;
	                        const int dd_points = method_points + 1;
	                        std::vector<double> nodes_method;
	                        nodes_method.reserve(static_cast<std::size_t>(method_points));
	                        nodes_method.push_back(0.0);
	                        double accum = 0.0;
	                        for (int j = 1; j < method_points; ++j) {
	                            accum += (j == 1) ? dt_step : historyDt(j - 2);
	                            nodes_method.push_back(-accum);
	                        }

	                        const auto a = math::finiteDifferenceWeights(/*derivative_order=*/deriv_order,
	                                                                     /*x0=*/0.0,
	                                                                     nodes_method);
	                        FE_THROW_IF(static_cast<int>(a.size()) != method_points, systems::InvalidStateException,
	                                    "TimeLoop: VSVO_BDF LTE weight size mismatch (method)");

	                        // Error constant for the derivative approximation on the polynomial t^{p1}.
	                        double c = 0.0;
	                        for (int j = 0; j < method_points; ++j) {
	                            c += a[static_cast<std::size_t>(j)] * powInt(nodes_method[static_cast<std::size_t>(j)], p1);
	                        }

	                        // dd_{p1} from dd_points states {u^{n+1}, u^n, ...}.
	                        std::vector<double> nodes_dd;
	                        nodes_dd.reserve(static_cast<std::size_t>(dd_points));
	                        nodes_dd.push_back(0.0);
	                        accum = 0.0;
	                        for (int j = 1; j < dd_points; ++j) {
	                            accum += (j == 1) ? dt_step : historyDt(j - 2);
	                            nodes_dd.push_back(-accum);
	                        }

	                        const auto w = math::finiteDifferenceWeights(/*derivative_order=*/p1, /*x0=*/0.0, nodes_dd);
	                        FE_THROW_IF(static_cast<int>(w.size()) != dd_points, systems::InvalidStateException,
	                                    "TimeLoop: VSVO_BDF LTE weight size mismatch (dd)");

	                        const double denom = factorial(p1);
                        FE_THROW_IF(!(denom > 0.0) || !std::isfinite(denom),
                                    systems::InvalidStateException,
                                    "TimeLoop: VSVO_BDF invalid factorial for LTE estimate");
	                        double dt_scale = dt_step;
	                        if (deriv_order >= 2) {
	                            dt_scale *= dt_step;
	                        }
	                        const double fac = dt_scale * c / denom;

                        const double atol = vsvo_controller->absTol();
                        const double rtol = vsvo_controller->relTol();
                        const auto u_np1 = history.uSpan();
                        FE_CHECK_ARG(u_np1.size() == history.uPrevSpan().size(), "TimeLoop: VSVO_BDF LTE size mismatch");

	                        double sum = 0.0;
	                        for (std::size_t i = 0; i < u_np1.size(); ++i) {
	                            // u^{n+1} coefficient
	                            double deriv = w[0] * static_cast<double>(u_np1[i]);

	                            // u^n .. history span(s)
	                            for (int j = 1; j < dd_points; ++j) {
	                                const auto uj = history.uPrevKSpan(j);
	                                deriv += w[static_cast<std::size_t>(j)] * static_cast<double>(uj[i]);
	                            }

                            const double lte = fac * deriv;
                            const double scale = atol + rtol * std::abs(static_cast<double>(u_np1[i]));
                            const double r = lte / scale;
                            sum += r * r;
                        }
                        const double n = (u_np1.empty() ? 1.0 : static_cast<double>(u_np1.size()));
                        return std::sqrt(sum / n);
                    };

                    auto computePredictor = [&](int p, double dt_step) {
                        auto dst = vsvo_pred->localSpan();
                        std::fill(dst.begin(), dst.end(), static_cast<Real>(0.0));

                        std::vector<double> nodes;
                        const int n_points = std::min(history.historyDepth(), p + 1);
                        nodes.reserve(static_cast<std::size_t>(n_points));
                        nodes.push_back(0.0);

                        const auto dt_hist = history.dtHistory();
                        const double dt_prev = (history.dtPrev() > 0.0 && std::isfinite(history.dtPrev()))
                            ? history.dtPrev()
                            : dt_step;

                        auto historyDt = [&](int idx) -> double {
                            if (idx < 0 || idx >= static_cast<int>(dt_hist.size())) {
                                return dt_prev;
                            }
                            const double v = dt_hist[static_cast<std::size_t>(idx)];
                            if (v > 0.0 && std::isfinite(v)) {
                                return v;
                            }
                            return dt_prev;
                        };

                        double accum = 0.0;
                        for (int j = 1; j < n_points; ++j) {
                            accum += historyDt(j - 1);
                            nodes.push_back(-accum);
                        }

                        const auto w = math::lagrangeWeights(dt_step, nodes);
                        FE_THROW_IF(static_cast<int>(w.size()) != n_points, systems::InvalidStateException,
                                    "TimeLoop: VSVO_BDF predictor weight mismatch");

                        for (int j = 0; j < n_points; ++j) {
                            const auto src = history.uPrevKSpan(j + 1);
                            const double alpha = w[static_cast<std::size_t>(j)];
                            FE_CHECK_ARG(src.size() == dst.size(), "TimeLoop: VSVO_BDF predictor size mismatch");
                            for (std::size_t i = 0; i < dst.size(); ++i) {
                                dst[i] += static_cast<Real>(alpha) * src[i];
                            }
                        }
                    };

                    computePredictor(order, dt);
                    copyVector(history.u(), *vsvo_pred);

                    auto integrator = vsvo_integrators[static_cast<std::size_t>(order)];
                    FE_CHECK_NOT_NULL(integrator.get(), "TimeLoop: VSVO_BDF integrator");
                    systems::TransientSystem transient_step(transient.system(), integrator);
                    nr = newton.solveStep(transient_step, linear, solve_time, history, workspace);

                    if (nr.converged) {
                        if (have_reference_solution) {
                            // Embedded reference estimate (first step only).
                            const double atol = vsvo_controller->absTol();
                            const double rtol = vsvo_controller->relTol();

                            const auto u = history.uSpan();
                            const auto ref = scratch_vec1->localSpan();
                            FE_CHECK_ARG(u.size() == ref.size(), "TimeLoop: VSVO_BDF size mismatch");

                            double sum = 0.0;
                            for (std::size_t i = 0; i < u.size(); ++i) {
                                const double scale = atol + rtol * std::abs(static_cast<double>(u[i]));
                                const double e = static_cast<double>(u[i] - ref[i]);
                                sum += (e / scale) * (e / scale);
                            }
                            const double denom = (u.empty() ? 1.0 : static_cast<double>(u.size()));
                            error_norm = std::sqrt(sum / denom);
                        } else {
                            error_norm = computeLTENorm(order, dt);
                            error_norm_low = computeLTENorm(order - 1, dt);
                            error_norm_high = computeLTENorm(order + 1, dt);
                        }
                    }
                    }
                } else if (options_.scheme == SchemeKind::TRBDF2) {
                    const double dt_saved = dt;
                    const double dt_prev_saved = history.dtPrev();
                    copyVector(*scratch_vec1, history.uPrev());
                    copyVector(*scratch_vec2, history.uPrev2());

                    bool restore_on_exit = true;
                    struct RestoreGuard {
                        TimeHistory& history;
                        double dt_saved;
                        double dt_prev_saved;
                        backends::GenericVector& saved_prev;
                        backends::GenericVector& saved_prev2;
                        bool& restore_on_exit;

                        ~RestoreGuard() noexcept
                        {
                            if (!restore_on_exit) {
                                return;
                            }
                            try {
                                history.setDt(dt_saved);
                                history.setPrevDt(dt_prev_saved);
                                copyVector(history.uPrev(), saved_prev);
                                copyVector(history.uPrev2(), saved_prev2);
                                history.resetCurrentToPrevious();
                            } catch (...) {
                                // Best-effort restoration: never throw from destructor.
                            }
                        }
                    } restore_guard{history, dt_saved, dt_prev_saved, *scratch_vec1, *scratch_vec2, restore_on_exit};

                    const double gamma = options_.trbdf2_gamma;
                    const double dt1 = gamma * dt;
                    const double dt2 = dt - dt1;
                    FE_THROW_IF(!(dt1 > 0.0) || !(dt2 > 0.0), systems::InvalidStateException, "TimeLoop: invalid TRBDF2 substep sizes");

                    // Stage 1: trapezoidal rule over dt1 (theta = 1/2).
                    history.setDt(dt1);
                    history.resetCurrentToPrevious();

                    const double stage1_time = t + dt1;
                    ImplicitStageSpec stage1;
                    stage1.integrator = bdf1;
                    stage1.weights.time_derivative = static_cast<Real>(1.0);
                    stage1.weights.non_time_derivative = static_cast<Real>(0.5);
                    stage1.solve_time = stage1_time;

                    ResidualAdditionSpec stage1_add;
                    stage1_add.integrator = bdf1;
                    stage1_add.weights.time_derivative = static_cast<Real>(0.0);
                    stage1_add.weights.non_time_derivative = static_cast<Real>(0.5);

                    systems::SystemStateView stage1_prev_state;
                    stage1_prev_state.time = history.time();
                    stage1_prev_state.dt = dt1;
                    stage1_prev_state.dt_prev = dt_prev_saved;
                    stage1_prev_state.u = scratch_vec1->localSpan();
                    stage1_prev_state.u_prev = scratch_vec1->localSpan();
                    stage1_prev_state.u_prev2 = scratch_vec2->localSpan();
                    stage1_prev_state.u_vector = scratch_vec1.get();
                    stage1_prev_state.u_prev_vector = scratch_vec1.get();
                    stage1_prev_state.u_prev2_vector = scratch_vec2.get();
                    stage1_prev_state.u_history = history.uHistorySpans();
                    stage1_prev_state.dt_history = history.dtHistory();

                    stage1_add.state = stage1_prev_state;
                    stage1.residual_addition = stage1_add;

                    nr = stages.solveImplicitStage(transient.system(), linear, history, workspace, stage1, scratch_vec0.get());

                    if (nr.converged) {
                        // Stage 2: BDF2 over dt2 using {u^{n+1}, u^{n+gamma}, u^n}.
                        history.setDt(dt2);
                        history.setPrevDt(dt1);

                        copyVector(history.uPrev2(), *scratch_vec1);      // u^{n}
                        copyVector(history.uPrev(), history.u());         // u^{n+gamma}
                        history.resetCurrentToPrevious();                 // initial guess = u^{n+gamma}

                        systems::TransientSystem transient_stage2(transient.system(), bdf2);
                        const double stage2_time = solve_time;
                        nr = newton.solveStep(transient_stage2, linear, stage2_time, history, workspace);

                        // Restore u^n so acceptStep shifts history correctly for the full step.
                        copyVector(history.uPrev(), *scratch_vec1);
                        // Restore u^{n-1} as well so deeper history (if present) shifts correctly.
                        copyVector(history.uPrev2(), *scratch_vec2);

                        // Restore dt so acceptStep advances the full step.
                        history.setDt(dt_saved);
                        restore_on_exit = false;
                    }
                } else {
                    FE_THROW(NotImplementedException, "TimeLoop: unsupported scheme");
                }
            } catch (const FEException&) {
                threw = true;
                caught_exception = std::current_exception();
                nr = NewtonReport{};
                nr.converged = false;
            }

            if (callbacks.on_nonlinear_done) {
                callbacks.on_nonlinear_done(history, nr);
            }

            auto make_step_attempt_info = [&](bool nonlinear_converged) {
                StepAttemptInfo info;
                info.time = t;
                info.t_end = t_end;
                info.dt = dt;
                info.dt_prev = dt_prev_step;
                info.step_index = step;
                info.attempt_index = attempt;
                info.scheme_order = scheme_order;
                info.nonlinear_converged = nonlinear_converged;
                info.newton = nr;
                info.error_norm = error_norm;
                info.error_norm_low = error_norm_low;
                info.error_norm_high = error_norm_high;
                return info;
            };

            using StateSyncPoint =
                NewtonOptions::StateSynchronizationPoint;
            StepCandidateRollbackGuard candidate_rollback_guard([&] {
                if (callbacks.on_step_candidate_discarded) {
                    callbacks.on_step_candidate_discarded(history);
                }
            });
            auto restoreAcceptedGeneratedState = [&]() {
                candidate_rollback_guard.discard();
                if (!options_.newton.synchronize_state) {
                    return;
                }

                // A nonlinear stage, candidate callback, or adaptive decision
                // may have left generated quadrature/curvature/active-set data
                // at a speculative state.  Rebuild it from u_n before retry or
                // terminal rejection.  The first callback may reconstruct
                // state-dependent constraints; project the already accepted
                // state at its own time before regenerating dependent data.
                auto restored_state = makeRestoredTimeStepStateView(
                    history, t, dt);
                options_.newton.synchronize_state(
                    restored_state,
                    StateSyncPoint::RestoredTimeStepState);

                const double accepted_dt =
                    history.dtPrev() > 0.0 ? history.dtPrev() : dt;
                transient.system().updateConstraints(t, accepted_dt);
                ensure_workspace_matches_sparsity();
                history.updateGhosts();
                if (!transient.system().constraints().empty()) {
                    transient.system().constraints().updateGhostsAndDistribute(
                        history.uPrev());
                }

                restored_state = makeRestoredTimeStepStateView(
                    history, t, dt);
                options_.newton.synchronize_state(
                    restored_state,
                    StateSyncPoint::RestoredProjectedTimeStepState);
            };

            bool accept_step = nr.converged;

            if (accept_step) {
                bool before_step_accept = true;
                if (callbacks.on_before_step_accept) {
                    candidate_rollback_guard.arm();
                    try {
                        before_step_accept =
                            callbacks.on_before_step_accept(history, nr);
                    } catch (...) {
                        const auto callback_failure =
                            std::current_exception();
                        restoreAcceptedGeneratedState();
                        std::rethrow_exception(callback_failure);
                    }
                }
                if (!before_step_accept) {
                    restoreAcceptedGeneratedState();
                    if (callbacks.on_step_rejected) {
                        callbacks.on_step_rejected(
                            history, StepRejectReason::ErrorTooLarge, nr);
                    }
                    if (!adaptive) {
                        FE_THROW(
                            FEException,
                            "TimeLoop: converged step rejected before accept");
                    }

                    const auto info = make_step_attempt_info(/*nonlinear_converged=*/true);
                    const auto decision =
                        options_.step_controller->onRejected(
                            info, StepRejectReason::ErrorTooLarge);
                    if (!decision.retry) {
                        report.success = false;
                        report.steps_taken = step;
                        report.final_time = history.time();
                        const std::string base =
                            decision.message.empty()
                                ? "TimeLoop: converged step rejected before accept"
                                : decision.message;
                        if (info.error_norm > 0.0 && std::isfinite(info.error_norm)) {
                            report.message = base + " (dt=" +
                                std::to_string(info.dt) + ", order=" +
                                std::to_string(info.scheme_order) +
                                ", error_norm=" +
                                std::to_string(info.error_norm) + ")";
                        } else {
                            report.message = base + " (dt=" +
                                std::to_string(info.dt) + ", order=" +
                                std::to_string(info.scheme_order) + ")";
                        }
                        return report;
                    }

                    const double new_dt = decision.next_dt;
                    FE_THROW_IF(!(new_dt > 0.0) || !std::isfinite(new_dt),
                                systems::InvalidStateException,
                                "TimeLoop: step controller returned invalid dt");
                    if (callbacks.on_dt_updated) {
                        callbacks.on_dt_updated(dt, new_dt, step, attempt);
                    }
                    dt = new_dt;
                    if (decision.next_order > 0) {
                        order = decision.next_order;
                    }
                    continue;
                }

                if (adaptive) {
                    const auto info = make_step_attempt_info(/*nonlinear_converged=*/true);

                    const auto decision = options_.step_controller->onAccepted(info);
                    if (!decision.accept) {
                        restoreAcceptedGeneratedState();
                        if (callbacks.on_step_rejected) {
                            callbacks.on_step_rejected(
                                history, StepRejectReason::ErrorTooLarge, nr);
                        }
                        if (!decision.retry) {
                            report.success = false;
                            report.steps_taken = step;
                            report.final_time = history.time();
                            const std::string base =
                                decision.message.empty()
                                    ? "TimeLoop: step rejected"
                                    : decision.message;
                            if (info.error_norm > 0.0 && std::isfinite(info.error_norm)) {
                                report.message =
                                    base + " (dt=" + std::to_string(info.dt) +
                                    ", order=" +
                                    std::to_string(info.scheme_order) +
                                    ", error_norm=" +
                                    std::to_string(info.error_norm) + ")";
                            } else {
                                report.message =
                                    base + " (dt=" + std::to_string(info.dt) +
                                    ", order=" +
                                    std::to_string(info.scheme_order) + ")";
                            }
                            return report;
                        }

                        const double new_dt = decision.next_dt;
                        FE_THROW_IF(!(new_dt > 0.0) || !std::isfinite(new_dt),
                                    systems::InvalidStateException,
                                    "TimeLoop: step controller returned invalid dt");
                        if (callbacks.on_dt_updated) {
                            callbacks.on_dt_updated(dt, new_dt, step, attempt);
                        }
                        dt = new_dt;
                        if (decision.next_order > 0) {
                            order = decision.next_order;
                        }
                        continue;
                    }

                    if (decision.next_dt > 0.0 && std::isfinite(decision.next_dt)) {
                        const double old = dt_next;
                        dt_next = decision.next_dt;
                        if (callbacks.on_dt_updated && old != dt_next) {
                            callbacks.on_dt_updated(old, dt_next, step, attempt);
                        }
                    }
                    if (decision.next_order > 0) {
                        order_next = decision.next_order;
                    }
                }

                const int temporal_order = transient.system().temporalOrder();
                if (temporal_order == 2 && history.hasSecondOrderState()) {
                    if (options_.scheme == SchemeKind::Newmark) {
                        const double beta = options_.newmark_beta;
                        const double gamma = options_.newmark_gamma;

                        auto u_np1 = history.uSpan();
                        const auto u_n = history.uPrevSpan();
                        auto v_n = history.uDotSpan();
                        auto a_n = history.uDDotSpan();

                        const double inv_beta = 1.0 / beta;
                        const double inv_dt = 1.0 / dt;
                        const double inv_beta_dt = inv_beta * inv_dt;
                        const double inv_beta_dt2 = inv_beta_dt * inv_dt;

                        const double a_c_a = (1.0 - 0.5 * inv_beta);
                        for (std::size_t i = 0; i < u_np1.size(); ++i) {
                            const Real u0 = u_n[i];
                            const Real v0 = v_n[i];
                            const Real a0 = a_n[i];
                            const Real u1 = u_np1[i];

                            const Real a1 = static_cast<Real>(inv_beta_dt2) * (u1 - u0 - static_cast<Real>(dt) * v0) +
                                static_cast<Real>(a_c_a) * a0;
                            const Real v1 = v0 + static_cast<Real>(dt) * (static_cast<Real>(1.0 - gamma) * a0 +
                                                                         static_cast<Real>(gamma) * a1);
                            a_n[i] = a1;
                            v_n[i] = v1;
                        }
                    } else if (options_.scheme == SchemeKind::GeneralizedAlpha) {
                        if (!ga2_params.has_value()) {
                            ga2_params = utils::generalizedAlphaSecondOrderFromRhoInf(options_.generalized_alpha_rho_inf);
                        }
                        const double beta = ga2_params->beta;
                        const double gamma = ga2_params->gamma;

                        auto u_np1 = history.uSpan();
                        const auto u_n = history.uPrevSpan();
                        auto v_n = history.uDotSpan();
                        auto a_n = history.uDDotSpan();

                        const double inv_beta = 1.0 / beta;
                        const double inv_dt = 1.0 / dt;
                        const double inv_beta_dt = inv_beta * inv_dt;
                        const double inv_beta_dt2 = inv_beta_dt * inv_dt;

                        const double a_c_a = (1.0 - 0.5 * inv_beta);
                        for (std::size_t i = 0; i < u_np1.size(); ++i) {
                            const Real u0 = u_n[i];
                            const Real v0 = v_n[i];
                            const Real a0 = a_n[i];
                            const Real u1 = u_np1[i];

                            const Real a1 = static_cast<Real>(inv_beta_dt2) * (u1 - u0 - static_cast<Real>(dt) * v0) +
                                static_cast<Real>(a_c_a) * a0;
                            const Real v1 = v0 + static_cast<Real>(dt) * (static_cast<Real>(1.0 - gamma) * a0 +
                                                                         static_cast<Real>(gamma) * a1);
                            a_n[i] = a1;
                            v_n[i] = v1;
                        }
                    } else if (used_collocation) {
                        const auto& so_data = getSecondOrderCollocationData(collocation_family_used, collocation_stages_used);

                        auto U_all = collocation.stage_values->localSpan();
                        FE_CHECK_ARG(U_all.size() == static_cast<std::size_t>(collocation_stages_used) * static_cast<std::size_t>(n_dofs),
                                     "TimeLoop: collocation stage_values size mismatch on accept");

                        const auto u_n = history.uPrevSpan();
                        const auto dv0 = collocation.dv0->localSpan();
                        FE_CHECK_ARG(u_n.size() == dv0.size(), "TimeLoop: collocation u_n/dv0 size mismatch");

                        auto scratch = collocation.stage_combination->localSpan();
                        FE_CHECK_ARG(scratch.size() == u_n.size(), "TimeLoop: collocation scratch size mismatch");

                        // v_{n+1} = p'(1)/dt, where p'(1) is expressed in terms of (u_n, dt*v_n, U_j).
                        std::fill(scratch.begin(), scratch.end(), static_cast<Real>(0.0));
                        for (std::size_t k = 0; k < scratch.size(); ++k) {
                            scratch[k] =
                                static_cast<Real>(so_data.du1_u0) * u_n[k] +
                                static_cast<Real>(so_data.du1_dv0) * dv0[k];
                        }
                        for (int j = 0; j < collocation_stages_used; ++j) {
                            const double cj = so_data.du1[static_cast<std::size_t>(j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < scratch.size(); ++k) {
                                scratch[k] += static_cast<Real>(cj) * Uj[k];
                            }
                        }
                        {
                            const double inv_dt = 1.0 / dt;
                            auto v_np1 = history.uDotSpan();
                            FE_CHECK_ARG(v_np1.size() == scratch.size(), "TimeLoop: collocation uDot size mismatch");
                            for (std::size_t k = 0; k < scratch.size(); ++k) {
                                v_np1[k] = static_cast<Real>(inv_dt) * scratch[k];
                            }
                        }

                        // a_{n+1} = p''(1)/dt^2.
                        std::fill(scratch.begin(), scratch.end(), static_cast<Real>(0.0));
                        for (std::size_t k = 0; k < scratch.size(); ++k) {
                            scratch[k] =
                                static_cast<Real>(so_data.ddu1_u0) * u_n[k] +
                                static_cast<Real>(so_data.ddu1_dv0) * dv0[k];
                        }
                        for (int j = 0; j < collocation_stages_used; ++j) {
                            const double cj = so_data.ddu1[static_cast<std::size_t>(j)];
                            const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                          static_cast<std::size_t>(n_dofs));
                            for (std::size_t k = 0; k < scratch.size(); ++k) {
                                scratch[k] += static_cast<Real>(cj) * Uj[k];
                            }
                        }
                        {
                            const double inv_dt2 = 1.0 / (dt * dt);
                            auto a_np1 = history.uDDotSpan();
                            FE_CHECK_ARG(a_np1.size() == scratch.size(), "TimeLoop: collocation uDDot size mismatch");
                            for (std::size_t k = 0; k < scratch.size(); ++k) {
                                a_np1[k] = static_cast<Real>(inv_dt2) * scratch[k];
                            }
                        }
                    }
                }

                const bool needs_endpoint_finalization =
                    (options_.scheme == SchemeKind::GeneralizedAlpha) ||
                    (used_collocation && collocation_family_used == CollocationFamily::Gauss);

                if (needs_endpoint_finalization) {
                    // The nonlinear solve for generalized-alpha (and
                    // non-endpoint Gauss collocation) leaves generated
                    // state at an intermediate stage.  Refresh endpoint
                    // topology first, apply its constraints to u_{n+1},
                    // and only then regenerate curvature/extension data.
                    // This occurs after all adaptive decisions and before
                    // the first irreversible commit, so it executes once
                    // for an accepted attempt and remains transactional.
                    if (options_.newton.synchronize_state) {
                        try {
                            const auto endpoint_state =
                                makeAcceptedTimeStepStateView(
                                    history, t + dt);
                            options_.newton.synchronize_state(
                                endpoint_state,
                                StateSyncPoint::EndpointCandidateState);
                        } catch (...) {
                            const auto endpoint_failure =
                                std::current_exception();
                            restoreAcceptedGeneratedState();
                            std::rethrow_exception(endpoint_failure);
                        }
                    }

                    transient.system().updateConstraints(t + dt, dt);
                    ensure_workspace_matches_sparsity();
                    history.updateGhosts();
                    if (!transient.system().constraints().empty()) {
                        // Do not apply endpoint-time inhomogeneities to
                        // u_n/u_{n-1}; those vectors represent their own
                        // accepted times.  Only the endpoint candidate is
                        // projected here.
                        transient.system().constraints().
                            updateGhostsAndDistribute(history.u());
                    }

                    if (generalized_alpha_first_order_rate_n_saved) {
                        FE_THROW_IF(
                            !ga1_params.has_value() ||
                                !history.hasUDotState(),
                            systems::InvalidStateException,
                            "TimeLoop: missing first-order generalized-alpha endpoint rate state");
                        const double gamma = ga1_params->gamma;
                        const double inv_gamma_dt = 1.0 / (gamma * dt);
                        const double c_old = (1.0 - gamma) / gamma;
                        const auto u_np1 = history.uSpan();
                        const auto u_n = history.uPrevSpan();
                        const auto rate_n =
                            generalized_alpha_rate_n->localSpan();
                        auto rate_np1 = history.uDotSpan();
                        FE_CHECK_ARG(
                            u_np1.size() == u_n.size() &&
                                u_np1.size() == rate_n.size() &&
                                u_np1.size() == rate_np1.size(),
                            "TimeLoop: generalized-alpha endpoint rate reconstruction size mismatch");
                        for (std::size_t i = 0; i < u_np1.size(); ++i) {
                            rate_np1[i] =
                                static_cast<Real>(inv_gamma_dt) *
                                    (u_np1[i] - u_n[i]) -
                                static_cast<Real>(c_old) * rate_n[i];
                        }
                        const auto dt_fields =
                            transient.system().timeDerivativeFields(
                                options_.newton.jacobian_op);
                        const auto nondt_dofs =
                            collectNonTimeDerivativeDofs(
                                transient.system(), dt_fields);
                        if (!nondt_dofs.empty()) {
                            zeroVectorEntries(nondt_dofs,
                                              history.uDot());
                        }
                    }

                    if (options_.newton.synchronize_state) {
                        try {
                            const auto projected_endpoint_state =
                                makeAcceptedTimeStepStateView(
                                    history, t + dt);
                            options_.newton.synchronize_state(
                                projected_endpoint_state,
                                StateSyncPoint::
                                    ProjectedEndpointCandidateState);
                        } catch (...) {
                            const auto endpoint_failure =
                                std::current_exception();
                            restoreAcceptedGeneratedState();
                            std::rethrow_exception(endpoint_failure);
                        }
                    }
                }

                if (monolithic_aux_stage_alpha_f.has_value()) {
                    const Real alpha_f = static_cast<Real>(*monolithic_aux_stage_alpha_f);
                    const Real gamma =
                        (ga1_params.has_value())
                            ? static_cast<Real>(ga1_params->gamma)
                            : Real(-1.0);
                    transient.system().finalizeMonolithicAuxiliaryStageState(
                        alpha_f,
                        gamma,
                        static_cast<Real>(dt),
                        static_cast<Real>(t + dt));
                }
                const auto accepted_time_step_state =
                    makeAcceptedTimeStepStateView(history, solve_time);
                // acceptGeometricNonlinearityState() is the first
                // irreversible acceptance operation: moving-mesh state
                // may be committed, its rollback backup is released, and
                // transaction callbacks may run.  All retry decisions are
                // complete at this point.  From this boundary onward a
                // fatal exception must retain the candidate rate state,
                // rather than rolling rates back independently of an
                // accepted or partially accepted system state.
                if (candidate_rollback_guard.armed()) {
                    try {
                        if (callbacks.on_step_commit_ready) {
                            callbacks.on_step_commit_ready(history);
                        }
                        candidate_rollback_guard.release();
                    } catch (...) {
                        const auto commit_failure = std::current_exception();
                        restoreAcceptedGeneratedState();
                        std::rethrow_exception(commit_failure);
                    }
                }
                attempt_rate_state.commit();
                transient.system().acceptGeometricNonlinearityState(
                    accepted_time_step_state,
                    systems::GeometricNonlinearityUpdatePoint::AcceptedTimeStep);
                transient.system().commitTimeStep();
                history.acceptStep(dt);
                if (temporal_order == 2 && options_.scheme == SchemeKind::VSVO_BDF && history.hasSecondOrderState()) {
                    (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                        history,
                        history.uDot().localSpan(),
                        history.uDDot().localSpan(),
                        /*overwrite_u_dot=*/true,
                        /*overwrite_u_ddot=*/true);
                    const auto& constraints = transient.system().constraints();
                    if (!constraints.empty()) {
                        constraints.distributeHomogeneous(history.uDot());
                        constraints.distributeHomogeneous(history.uDDot());
                    }
                }
                if (callbacks.on_step_accepted) {
                    callbacks.on_step_accepted(history);
                }

                // Interface-tracking MPC constraints (e.g. small-cut
                // aggregation) re-classify their slave sets in the
                // accepted-step refresh, which runs inside the
                // on_step_accepted callback above. A DOF that changed
                // status during this step carries a finite-difference rate
                // polluted by the jump between its free trajectory and the
                // extension trajectory scaled by 1/(gamma*dt); the
                // alpha_m-weighted mass term broadcasts that pulse into
                // neighboring momentum rows on the next stage solve, which
                // shows up as a band-localized velocity-error floor under
                // h- and dt-refinement (open-vessel MMS). Re-impose the
                // constraint-consistent state on master-bearing lines
                // only: values follow the extension, rates follow the
                // masters' rates (the exact derivative of a constraint
                // with time-constant coefficients). Dirichlet lines are
                // untouched so their finite-difference rates keep carrying
                // g_dot (see zeroConstrainedRatesRequested above).
                if (!mpcAcceptedStateDistributeDisabled()) {
                    const auto& accepted_constraints =
                        transient.system().constraints();
                    if (!accepted_constraints.empty() &&
                        accepted_constraints.hasMasterBearingLines()) {
                        history.updateGhosts();
                        accepted_constraints.distributeMasterBearing(history.u());
                        accepted_constraints.distributeMasterBearing(history.uPrev());
                        if (history.hasUDotState()) {
                            accepted_constraints.distributeMasterBearingHomogeneous(
                                history.uDot());
                        }
                        if (history.hasUDDotState()) {
                            accepted_constraints.distributeMasterBearingHomogeneous(
                                history.uDDot());
                        }
                    }
                }
                accepted = true;
                break;
            }

            if (!adaptive) {
                restoreAcceptedGeneratedState();
                if (threw && caught_exception) {
                    try {
                        std::rethrow_exception(caught_exception);
                    } catch (FEException& e) {
                        e.add_context("TimeLoop: nonlinear solve threw an exception");
                        throw;
                    }
                }
                FE_THROW(FEException, "TimeLoop: nonlinear solve did not converge");
            }

            restoreAcceptedGeneratedState();
            if (callbacks.on_step_rejected) {
                callbacks.on_step_rejected(history, StepRejectReason::NonlinearSolveFailed, nr);
            }

            const auto info = make_step_attempt_info(/*nonlinear_converged=*/false);

            const auto decision = options_.step_controller->onRejected(info, StepRejectReason::NonlinearSolveFailed);
            if (!decision.retry) {
                report.success = false;
                report.steps_taken = step;
                report.final_time = history.time();
                report.message = decision.message.empty() ? "TimeLoop: step rejected" : decision.message;
                // Adaptive retry policy must not erase the exception that made
                // the nonlinear attempt fail.  Without this, an assembly or
                // backend exception is reduced to the generic "nonlinear solve
                // failed" message after the final retry, which makes a
                // production failure impossible to diagnose from its log.
                if (threw && caught_exception) {
                    try {
                        std::rethrow_exception(caught_exception);
                    } catch (const std::exception& error) {
                        report.message += " (exception: ";
                        report.message += error.what();
                        report.message += ")";
                    } catch (...) {
                        report.message += " (exception: non-standard exception)";
                    }
                }
                return report;
            }

            const double new_dt = decision.next_dt;
            FE_THROW_IF(!(new_dt > 0.0) || !std::isfinite(new_dt), systems::InvalidStateException,
                        "TimeLoop: step controller returned invalid dt");
            if (callbacks.on_dt_updated) {
                callbacks.on_dt_updated(dt, new_dt, step, attempt);
            }
            dt = new_dt;
            if (decision.next_order > 0) {
                order = decision.next_order;
            }
        }

        if (!accepted) {
            report.success = false;
            report.steps_taken = step;
            report.final_time = history.time();
            report.message = "TimeLoop: step failed after retries";
            return report;
        }
    }

    // If the loop exits because step == max_steps, the final accepted step may
    // have advanced time exactly to t_end. Handle that edge case explicitly.
    const double t = history.time();
    if (t + time_tol >= t_end) {
        report.success = true;
        report.steps_taken = options_.max_steps;
        report.final_time = t_end;
        history.setTime(t_end);
        return report;
    }

    report.success = false;
    report.steps_taken = options_.max_steps;
    report.final_time = t;
    report.message = "TimeLoop: max_steps exceeded";
    return report;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
