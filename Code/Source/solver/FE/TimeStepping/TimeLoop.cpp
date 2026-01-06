/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/TimeLoop.h"

#include "Core/FEException.h"
#include "Math/FiniteDifference.h"
#include "Sparsity/SparsityPattern.h"
#include "Systems/SystemsExceptions.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/CollocationMethods.h"
#include "TimeStepping/MultiStageScheme.h"
#include "TimeStepping/NewmarkBeta.h"
#include "TimeStepping/TimeSteppingUtils.h"
#include "TimeStepping/VSVO_BDF_Controller.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

namespace {

void copyVector(backends::GenericVector& dst, const backends::GenericVector& src)
{
    auto d = dst.localSpan();
    auto s = src.localSpan();
    FE_CHECK_ARG(d.size() == s.size(), "TimeLoop: vector size mismatch");
    std::copy(s.begin(), s.end(), d.begin());
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

    // Ensure time-history vectors use the same backend layout as the solver workspace.
    // For backends like FSILS, vectors created before any matrix exists may not share
    // the matrix's internal ordering, which would corrupt updates like u <- u - du.
    history.repack(factory);

    if (options_.scheme == SchemeKind::VSVO_BDF && history.stepIndex() > 0) {
        // Restart sanity check: variable-step schemes require dtHistory() to match the provided
        // displacement history. Avoid silently fabricating older dt values via primeDtHistory().
        const int required_dt_history = std::min(history.stepIndex(), history.historyDepth());
        FE_THROW_IF(!history.dtHistoryIsValid(required_dt_history),
                    InvalidArgumentException,
                    "TimeLoop: VSVO_BDF restart requires a consistent dtHistory() (use TimeHistory::setDtHistory)");
    }
    history.primeDtHistory(history.dtPrev() > 0.0 ? history.dtPrev() : history.dt());

    auto scratch_vec0 = factory.createVector(n_dofs);
    auto scratch_vec1 = factory.createVector(n_dofs);
    auto scratch_vec2 = factory.createVector(n_dofs);
    FE_CHECK_NOT_NULL(scratch_vec0.get(), "TimeLoop scratch_vec0");
    FE_CHECK_NOT_NULL(scratch_vec1.get(), "TimeLoop scratch_vec1");
    FE_CHECK_NOT_NULL(scratch_vec2.get(), "TimeLoop scratch_vec2");

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
                constraints.distributeHomogeneous(reinterpret_cast<double*>(u_prev.data()),
                                                  static_cast<GlobalIndex>(u_prev.size()));
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
                auto v = history.uDotSpan();
                constraints.distributeHomogeneous(reinterpret_cast<double*>(v.data()),
                                                  static_cast<GlobalIndex>(v.size()));
            }
            if (overwrite_u_ddot) {
                auto a = history.uDDotSpan();
                constraints.distributeHomogeneous(reinterpret_cast<double*>(a.data()),
                                                  static_cast<GlobalIndex>(a.size()));
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

        history.updateGhosts();
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
                            constraints.distributeHomogeneous(reinterpret_cast<double*>(w.data()),
                                                              static_cast<GlobalIndex>(w.size()));
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
                            constraints.distributeHomogeneous(reinterpret_cast<double*>(w.data()),
                                                              static_cast<GlobalIndex>(w.size()));
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
                            constraints.distributeHomogeneous(reinterpret_cast<double*>(w.data()),
                                                              static_cast<GlobalIndex>(w.size()));
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

            const bool abs_ok = rep.residual_norm <= options_.newton.abs_tolerance;
            const bool rel_ok = (options_.newton.rel_tolerance <= 0.0)
                ? true
                : (rep.residual_norm0 > 0.0
                       ? (rep.residual_norm / rep.residual_norm0 <= options_.newton.rel_tolerance)
                       : abs_ok);
            if (abs_ok && rel_ok) {
                rep.converged = true;
                rep.iterations = it;
                break;
            }

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

    const double time_tol = 100.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(t_end));

    const bool adaptive = static_cast<bool>(options_.step_controller);
    const int max_retries = adaptive ? std::max(0, options_.step_controller->maxRetries()) : 0;
    double dt_next = options_.dt;

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
            if (remaining0 < dt) {
                dt = remaining0;
            }
        }

        bool accepted = false;
        NewtonReport nr;

        for (int attempt = 0; attempt <= max_retries; ++attempt) {
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
                if (remaining < dt) {
                    dt = remaining;
                }
            }

            history.setDt(dt);
            history.resetCurrentToPrevious();
            if (callbacks.on_step_start) {
                callbacks.on_step_start(history);
            }

            const double dt_prev_step = history.dtPrev();

            const double solve_time = t + dt;
            transient.system().beginTimeStep();

            int scheme_order = 0;
            double error_norm = -1.0;
            double error_norm_low = -1.0;
            double error_norm_high = -1.0;
            bool used_collocation = false;
            CollocationFamily collocation_family_used = CollocationFamily::Gauss;
            int collocation_stages_used = 0;

            bool threw = false;
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
                        history.updateGhosts();

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

                        // Ensure uDot storage exists and is initialized before the stage solve.
                        const bool had_u_dot = history.hasUDotState();
                        history.ensureSecondOrderState(factory);
                        if (!had_u_dot) {
                            (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                                history,
                                history.uDot().localSpan(),
                                history.uDDot().localSpan(),
                                /*overwrite_u_dot=*/true,
                                /*overwrite_u_ddot=*/false);
                        }
                        const auto& constraints = transient.system().constraints();
                        if (!constraints.empty()) {
                            auto v = history.uDotSpan();
                            constraints.distributeHomogeneous(reinterpret_cast<double*>(v.data()),
                                                              static_cast<GlobalIndex>(v.size()));
                        }

                        // Ensure (u_n, uDot_n) values are ghost-consistent before constructing constants.
                        history.updateGhosts();

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
                        nr = newton.solveStep(transient_stage, linear, stage_time, history, workspace);
                        if (nr.converged) {
                            const double inv_af = 1.0 / ga1_params->alpha_f;
                            const double c_prev = (ga1_params->alpha_f - 1.0) * inv_af;
                            auto cur = history.uSpan();
                            const auto prev = history.uPrevSpan();
                            FE_CHECK_ARG(cur.size() == prev.size(), "TimeLoop: generalized-alpha size mismatch");
                            for (std::size_t i = 0; i < cur.size(); ++i) {
                                cur[i] = static_cast<Real>(inv_af) * cur[i] + static_cast<Real>(c_prev) * prev[i];
                            }

                            // Update uDot_{n+1} (stored as TimeHistory::uDot) for use by later stages
                            // and end-of-step finalization.
                            const double gamma = ga1_params->gamma;
                            const double inv_gamma_dt = 1.0 / (gamma * dt);
                            const double c_old = (1.0 - gamma) / gamma;
                            auto v = history.uDotSpan();
                            FE_CHECK_ARG(v.size() == cur.size(), "TimeLoop: generalized-alpha uDot size mismatch");
                            for (std::size_t i = 0; i < cur.size(); ++i) {
                                const Real v_n = v[i];
                                v[i] = static_cast<Real>(inv_gamma_dt) * (cur[i] - prev[i]) -
                                    static_cast<Real>(c_old) * v_n;
                            }
                            if (!constraints.empty()) {
                                constraints.distributeHomogeneous(reinterpret_cast<double*>(v.data()),
                                                                  static_cast<GlobalIndex>(v.size()));
                            }
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
                        history.updateGhosts();

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

                            history.updateGhosts();

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
                nr = NewtonReport{};
                nr.converged = false;
            }

            if (callbacks.on_nonlinear_done) {
                callbacks.on_nonlinear_done(history, nr);
            }

            if (nr.converged) {
                if (adaptive) {
                    StepAttemptInfo info;
                    info.time = t;
                    info.t_end = t_end;
                    info.dt = dt;
                    info.dt_prev = dt_prev_step;
                    info.step_index = step;
                    info.attempt_index = attempt;
                    info.scheme_order = scheme_order;
                    info.nonlinear_converged = true;
                    info.newton = nr;
                    info.error_norm = error_norm;
                    info.error_norm_low = error_norm_low;
                    info.error_norm_high = error_norm_high;

	                    const auto decision = options_.step_controller->onAccepted(info);
	                    if (!decision.accept) {
	                        if (callbacks.on_step_rejected) {
	                            callbacks.on_step_rejected(history, StepRejectReason::ErrorTooLarge, nr);
	                        }
	                        if (!decision.retry) {
	                            report.success = false;
	                            report.steps_taken = step;
	                            report.final_time = history.time();
	                            const std::string base = decision.message.empty() ? "TimeLoop: step rejected" : decision.message;
	                            if (info.error_norm > 0.0 && std::isfinite(info.error_norm)) {
	                                report.message = base + " (dt=" + std::to_string(info.dt) +
	                                    ", order=" + std::to_string(info.scheme_order) +
	                                    ", error_norm=" + std::to_string(info.error_norm) + ")";
	                            } else {
	                                report.message = base + " (dt=" + std::to_string(info.dt) +
	                                    ", order=" + std::to_string(info.scheme_order) + ")";
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

                const bool needs_final_state_commit =
                    (options_.scheme == SchemeKind::GeneralizedAlpha) ||
                    (used_collocation && collocation_family_used == CollocationFamily::Gauss);

                if (needs_final_state_commit) {
                    // Ensure stateful kernels (MaterialStateProvider / GlobalKernelStateProvider) are committed
                    // for the accepted end-of-step state at t_{n+1}, even when the scheme's nonlinear solve was
                    // performed at an intermediate stage time (e.g., generalized-α, Gauss collocation).
                    history.updateGhosts();

                    // Save displacement history since we overwrite the first two slots with
                    // end-state constants for dt(u) and dt(u,2).
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

                    const auto u_np1 = history.uSpan();
                    auto u_prev = history.uPrev().localSpan();
                    auto u_prev2 = history.uPrev2().localSpan();

                    if (temporal_order == 1) {
                        auto dt_u_dot = scratch_vec0->localSpan();
                        FE_CHECK_ARG(dt_u_dot.size() == u_np1.size(), "TimeLoop: dt*uDot scratch size mismatch");
                        std::fill(dt_u_dot.begin(), dt_u_dot.end(), static_cast<Real>(0.0));

                        if (options_.scheme == SchemeKind::GeneralizedAlpha) {
                            // Use the stored uDot_{n+1} to build a BDF1-consistent end-state history:
                            // dt(u) = (u_{n+1} - u_prev)/dt  =>  u_prev = u_{n+1} - dt*uDot_{n+1}.
                            FE_THROW_IF(!history.hasUDotState(), systems::InvalidStateException,
                                        "TimeLoop: missing uDot for generalized-alpha final-state commit");
                            const auto v_np1 = history.uDotSpan();
                            FE_CHECK_ARG(v_np1.size() == dt_u_dot.size(), "TimeLoop: generalized-alpha uDot size mismatch");
                            for (std::size_t k = 0; k < dt_u_dot.size(); ++k) {
                                dt_u_dot[k] = static_cast<Real>(dt) * v_np1[k];
                            }
                        } else if (used_collocation && collocation_family_used == CollocationFamily::Gauss) {
                            const auto& method = getCollocationMethod(collocation_family_used, collocation_stages_used);
                            FE_THROW_IF(method.stages != collocation_stages_used, systems::InvalidStateException,
                                        "TimeLoop: collocation stages mismatch on final-state commit");

                            std::vector<double> nodes;
                            nodes.reserve(static_cast<std::size_t>(method.stages + 1));
                            nodes.push_back(0.0);
                            for (int j = 0; j < method.stages; ++j) {
                                nodes.push_back(method.c[static_cast<std::size_t>(j)]);
                            }

                            const auto w = math::finiteDifferenceWeights(/*derivative_order=*/1, /*x0=*/1.0, nodes);
                            FE_THROW_IF(static_cast<int>(w.size()) != method.stages + 1, systems::InvalidStateException,
                                        "TimeLoop: collocation end-derivative weight mismatch");

                            const auto u_n = history.uPrevSpan();
                            FE_CHECK_ARG(u_n.size() == dt_u_dot.size(), "TimeLoop: collocation u_n size mismatch");
                            for (std::size_t k = 0; k < dt_u_dot.size(); ++k) {
                                dt_u_dot[k] += static_cast<Real>(w[0]) * u_n[k];
                            }

                            const auto U_all = collocation.stage_values->localSpan();
                            FE_CHECK_ARG(U_all.size() == static_cast<std::size_t>(method.stages) * static_cast<std::size_t>(n_dofs),
                                         "TimeLoop: collocation stage_values size mismatch on final-state commit");
                            for (int j = 0; j < method.stages; ++j) {
                                const double cj = w[static_cast<std::size_t>(j + 1)];
                                const auto Uj = U_all.subspan(static_cast<std::size_t>(j) * static_cast<std::size_t>(n_dofs),
                                                              static_cast<std::size_t>(n_dofs));
                                for (std::size_t k = 0; k < dt_u_dot.size(); ++k) {
                                    dt_u_dot[k] += static_cast<Real>(cj) * Uj[k];
                                }
                            }
                        } else {
                            FE_THROW(NotImplementedException, "TimeLoop: missing dt*uDot end-state reconstruction");
                        }

                        FE_CHECK_ARG(u_prev.size() == u_np1.size(), "TimeLoop: u_prev size mismatch");
                        for (std::size_t k = 0; k < u_prev.size(); ++k) {
                            u_prev[k] = u_np1[k] - dt_u_dot[k];
                        }
                    } else if (temporal_order == 2) {
                        FE_THROW_IF(!history.hasSecondOrderState(), systems::InvalidStateException,
                                    "TimeLoop: missing (uDot,uDDot) for 2nd-order final-state commit");
                        const auto v_np1 = history.uDotSpan();
                        const auto a_np1 = history.uDDotSpan();
                        FE_CHECK_ARG(v_np1.size() == u_np1.size(), "TimeLoop: uDot size mismatch on final-state commit");
                        FE_CHECK_ARG(a_np1.size() == u_np1.size(), "TimeLoop: uDDot size mismatch on final-state commit");
                        for (std::size_t k = 0; k < u_prev.size(); ++k) {
                            u_prev[k] = u_np1[k] - static_cast<Real>(dt) * v_np1[k];
                            u_prev2[k] = u_np1[k] - static_cast<Real>(2.0 * dt) * v_np1[k] +
                                static_cast<Real>(dt * dt) * a_np1[k];
                        }
                    }

                    systems::TransientSystem transient_finalize(transient.system(), bdf1);
                    systems::AssemblyRequest req_finalize;
                    req_finalize.op = options_.newton.residual_op;
                    req_finalize.want_vector = true;
                    req_finalize.zero_outputs = true;

                    scratch_vec0->zero();
                    auto out_view = scratch_vec0->createAssemblyView();
                    FE_CHECK_NOT_NULL(out_view.get(), "TimeLoop: final-state residual view");

                    systems::SystemStateView state;
                    state.time = solve_time;
                    state.dt = dt;
                    state.dt_prev = history.dtPrev();
                    state.u = history.uSpan();
                    state.u_prev = history.uPrevSpan();
                    state.u_prev2 = history.uPrev2Span();
                    state.u_history = history.uHistorySpans();
                    state.dt_history = history.dtHistory();

                    transient.system().beginTimeStep();
                    (void)transient_finalize.assemble(req_finalize, state, nullptr, out_view.get());
                }

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
	                        auto v = history.uDotSpan();
	                        constraints.distributeHomogeneous(reinterpret_cast<double*>(v.data()),
	                                                          static_cast<GlobalIndex>(v.size()));
	                        auto a = history.uDDotSpan();
	                        constraints.distributeHomogeneous(reinterpret_cast<double*>(a.data()),
	                                                          static_cast<GlobalIndex>(a.size()));
	                    }
	                }
	                if (callbacks.on_step_accepted) {
	                    callbacks.on_step_accepted(history);
	                }

                accepted = true;
                break;
            }

            if (!adaptive) {
                FE_THROW_IF(threw, FEException, "TimeLoop: nonlinear solve threw an exception");
                FE_THROW(FEException, "TimeLoop: nonlinear solve did not converge");
            }

            if (callbacks.on_step_rejected) {
                callbacks.on_step_rejected(history, StepRejectReason::NonlinearSolveFailed, nr);
            }

            StepAttemptInfo info;
            info.time = t;
            info.t_end = t_end;
            info.dt = dt;
            info.dt_prev = dt_prev_step;
            info.step_index = step;
            info.attempt_index = attempt;
            info.scheme_order = scheme_order;
            info.nonlinear_converged = false;
            info.newton = nr;
            info.error_norm = error_norm;
            info.error_norm_low = error_norm_low;
            info.error_norm_high = error_norm_high;

            const auto decision = options_.step_controller->onRejected(info, StepRejectReason::NonlinearSolveFailed);
            if (!decision.retry) {
                report.success = false;
                report.steps_taken = step;
                report.final_time = history.time();
                report.message = decision.message.empty() ? "TimeLoop: step rejected" : decision.message;
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
