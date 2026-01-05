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
                .history_rate_order = 2});
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
    history.primeDtHistory(history.dtPrev() > 0.0 ? history.dtPrev() : history.dt());

    auto scratch_vec0 = factory.createVector(n_dofs);
    auto scratch_vec1 = factory.createVector(n_dofs);
    auto scratch_vec2 = factory.createVector(n_dofs);
    FE_CHECK_NOT_NULL(scratch_vec0.get(), "TimeLoop scratch_vec0");
    FE_CHECK_NOT_NULL(scratch_vec1.get(), "TimeLoop scratch_vec1");
    FE_CHECK_NOT_NULL(scratch_vec2.get(), "TimeLoop scratch_vec2");

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

    struct CollocationMethod {
        int stages{0};
        int order{0};
        std::vector<double> c{};
        std::vector<double> ainv{};     // row-major (stages x stages)
        std::vector<double> row_sums{}; // sum_j ainv[i,j]
        std::vector<double> final_w{};  // u_{n+1} = u_n + sum_j final_w[j] * (U_j - u_n)

        bool stiffly_accurate{false};
        int final_stage{0};
    };

    enum class CollocationFamily : std::uint8_t {
        RadauIIA,
        Gauss
    };

    auto legendreWithDerivative = [](int n, double x) -> std::pair<double, double> {
        double p0 = 1.0;
        double p1 = x;

        if (n == 0) {
            return {p0, 0.0};
        }
        if (n == 1) {
            return {p1, p0};
        }

        for (int k = 2; k <= n; ++k) {
            const double pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / static_cast<double>(k);
            p0 = p1;
            p1 = pk;
        }

        const double denom = 1.0 - x * x;
        FE_THROW_IF(denom == 0.0, InvalidArgumentException,
                    "TimeLoop: Legendre derivative singularity at |x|=1");
        const double dp = static_cast<double>(n) / denom * (p0 - x * p1);
        return {p1, dp};
    };

    auto gaussNodesMinusOneOne = [&](int n) -> std::vector<double> {
        FE_THROW_IF(n <= 0, InvalidArgumentException, "TimeLoop: Gauss nodes require n > 0");
        FE_THROW_IF(n > 64, InvalidArgumentException, "TimeLoop: Gauss nodes n too large (max 64)");

        const int m = (n + 1) / 2;
        std::vector<double> nodes(static_cast<std::size_t>(n), 0.0);

        const double pi = std::acos(-1.0);
        constexpr double tol = 1e-14;
        for (int i = 0; i < m; ++i) {
            double z = std::cos(pi * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));
            double z_prev = std::numeric_limits<double>::max();
            for (int it = 0; it < 64; ++it) {
                if (std::abs(z - z_prev) <= tol) {
                    break;
                }
                z_prev = z;
                const auto [P, dP] = legendreWithDerivative(n, z);
                FE_THROW_IF(dP == 0.0, InvalidArgumentException, "TimeLoop: Gauss node Newton derivative is zero");
                z = z_prev - P / dP;
            }

            nodes[static_cast<std::size_t>(i)] = -z;
            nodes[static_cast<std::size_t>(n - 1 - i)] = z;
        }

        std::sort(nodes.begin(), nodes.end());
        return nodes;
    };

    auto gaussNodesUnit = [&](int stages) -> std::vector<double> {
        const auto x = gaussNodesMinusOneOne(stages);
        std::vector<double> c;
        c.reserve(x.size());
        for (double xi : x) {
            c.push_back(0.5 * (xi + 1.0));
        }
        return c;
    };

    auto radauIIANodesUnit = [&](int stages) -> std::vector<double> {
        FE_THROW_IF(stages <= 0, InvalidArgumentException, "TimeLoop: Radau IIA requires stages > 0");
        if (stages == 1) {
            return {1.0};
        }

        const auto guesses = gaussNodesMinusOneOne(stages - 1);
        std::vector<double> roots;
        roots.reserve(static_cast<std::size_t>(stages));

        constexpr double tol = 1e-14;
        for (double z0 : guesses) {
            double z = z0;
            double z_prev = std::numeric_limits<double>::max();
            for (int it = 0; it < 64; ++it) {
                if (std::abs(z - z_prev) <= tol) {
                    break;
                }
                z_prev = z;
                const auto [Ps, dPs] = legendreWithDerivative(stages, z);
                const auto [Ps1, dPs1] = legendreWithDerivative(stages - 1, z);
                const double f = Ps - Ps1;
                const double df = dPs - dPs1;
                FE_THROW_IF(df == 0.0, InvalidArgumentException, "TimeLoop: Radau node Newton derivative is zero");
                z = z_prev - f / df;
            }
            roots.push_back(z);
        }
        std::sort(roots.begin(), roots.end());
        roots.push_back(1.0);

        std::vector<double> c;
        c.reserve(roots.size());
        for (double xi : roots) {
            c.push_back(0.5 * (xi + 1.0));
        }
        return c;
    };

    auto invertDenseMatrix = [&](const std::vector<double>& A, int n) -> std::vector<double> {
        FE_THROW_IF(n <= 0, InvalidArgumentException, "TimeLoop: invalid dense matrix size");
        FE_THROW_IF(static_cast<int>(A.size()) != n * n, InvalidArgumentException,
                    "TimeLoop: dense matrix size mismatch");

        std::vector<double> M = A;
        std::vector<double> inv(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            inv[static_cast<std::size_t>(i * n + i)] = 1.0;
        }

        auto rowSwap = [&](std::vector<double>& mat, int r1, int r2) {
            for (int j = 0; j < n; ++j) {
                std::swap(mat[static_cast<std::size_t>(r1 * n + j)],
                          mat[static_cast<std::size_t>(r2 * n + j)]);
            }
        };

        for (int k = 0; k < n; ++k) {
            int piv = k;
            double piv_abs = std::abs(M[static_cast<std::size_t>(k * n + k)]);
            for (int r = k + 1; r < n; ++r) {
                const double a = std::abs(M[static_cast<std::size_t>(r * n + k)]);
                if (a > piv_abs) {
                    piv_abs = a;
                    piv = r;
                }
            }
            FE_THROW_IF(piv_abs == 0.0, InvalidArgumentException, "TimeLoop: singular dense matrix");
            if (piv != k) {
                rowSwap(M, piv, k);
                rowSwap(inv, piv, k);
            }

            const double diag = M[static_cast<std::size_t>(k * n + k)];
            const double inv_diag = 1.0 / diag;
            for (int j = 0; j < n; ++j) {
                M[static_cast<std::size_t>(k * n + j)] *= inv_diag;
                inv[static_cast<std::size_t>(k * n + j)] *= inv_diag;
            }

            for (int r = 0; r < n; ++r) {
                if (r == k) continue;
                const double fac = M[static_cast<std::size_t>(r * n + k)];
                if (fac == 0.0) continue;
                for (int j = 0; j < n; ++j) {
                    M[static_cast<std::size_t>(r * n + j)] -= fac * M[static_cast<std::size_t>(k * n + j)];
                    inv[static_cast<std::size_t>(r * n + j)] -= fac * inv[static_cast<std::size_t>(k * n + j)];
                }
            }
        }

        return inv;
    };

    auto buildCollocationMethod = [&](CollocationFamily family, int stages) -> CollocationMethod {
        FE_THROW_IF(stages <= 0, InvalidArgumentException, "TimeLoop: invalid collocation stage count");

        CollocationMethod method;
        method.stages = stages;
        method.order = (family == CollocationFamily::Gauss) ? 2 * stages : 2 * stages - 1;
        method.stiffly_accurate = (family == CollocationFamily::RadauIIA);
        method.final_stage = (family == CollocationFamily::RadauIIA) ? (stages - 1) : 0;

        method.c = (family == CollocationFamily::Gauss)
            ? gaussNodesUnit(stages)
            : radauIIANodesUnit(stages);

        auto lagrangeCoeff = [&](int j) -> std::vector<double> {
            std::vector<double> coeff(static_cast<std::size_t>(stages), 0.0);
            coeff[0] = 1.0;
            int deg = 0;
            const double cj = method.c[static_cast<std::size_t>(j)];

            for (int m = 0; m < stages; ++m) {
                if (m == j) continue;
                const double cm = method.c[static_cast<std::size_t>(m)];
                const double denom = cj - cm;
                FE_THROW_IF(denom == 0.0, InvalidArgumentException, "TimeLoop: duplicate collocation nodes");

                std::vector<double> next(static_cast<std::size_t>(stages), 0.0);
                for (int k = 0; k <= deg; ++k) {
                    next[static_cast<std::size_t>(k)] += (-cm / denom) * coeff[static_cast<std::size_t>(k)];
                    next[static_cast<std::size_t>(k + 1)] += (1.0 / denom) * coeff[static_cast<std::size_t>(k)];
                }
                coeff = std::move(next);
                ++deg;
            }
            return coeff;
        };

        std::vector<double> A(static_cast<std::size_t>(stages) * static_cast<std::size_t>(stages), 0.0);
        std::vector<double> b(static_cast<std::size_t>(stages), 0.0);

        for (int j = 0; j < stages; ++j) {
            const auto coeff = lagrangeCoeff(j);

            auto integratePoly = [&](double x) -> double {
                double sum = 0.0;
                double xpow = x;
                for (int k = 0; k < stages; ++k) {
                    sum += coeff[static_cast<std::size_t>(k)] * xpow / static_cast<double>(k + 1);
                    xpow *= x;
                }
                return sum;
            };

            b[static_cast<std::size_t>(j)] = integratePoly(1.0);
            for (int i = 0; i < stages; ++i) {
                const double ci = method.c[static_cast<std::size_t>(i)];
                A[static_cast<std::size_t>(i * stages + j)] = integratePoly(ci);
            }
        }

        method.ainv = invertDenseMatrix(A, stages);
        method.row_sums.resize(static_cast<std::size_t>(stages), 0.0);
        for (int i = 0; i < stages; ++i) {
            double sum = 0.0;
            for (int j = 0; j < stages; ++j) {
                sum += method.ainv[static_cast<std::size_t>(i * stages + j)];
            }
            method.row_sums[static_cast<std::size_t>(i)] = sum;
        }

        if (!method.stiffly_accurate) {
            method.final_w.resize(static_cast<std::size_t>(stages), 0.0);
            for (int j = 0; j < stages; ++j) {
                double sum = 0.0;
                for (int i = 0; i < stages; ++i) {
                    sum += b[static_cast<std::size_t>(i)] * method.ainv[static_cast<std::size_t>(i * stages + j)];
                }
                method.final_w[static_cast<std::size_t>(j)] = sum;
            }
        }

        return method;
    };

    std::unordered_map<int, CollocationMethod> collocation_gauss{};
    std::unordered_map<int, CollocationMethod> collocation_radau{};

    auto getCollocationMethod = [&](CollocationFamily family, int stages) -> const CollocationMethod& {
        auto& cache = (family == CollocationFamily::Gauss) ? collocation_gauss : collocation_radau;
        auto it = cache.find(stages);
        if (it != cache.end()) {
            return it->second;
        }
        auto [ins_it, inserted] = cache.emplace(stages, buildCollocationMethod(family, stages));
        FE_CHECK_ARG(inserted, "TimeLoop: failed to cache collocation method");
        return ins_it->second;
    };

    struct SecondOrderCollocationData {
        int stages{0};
        int n_constraints{0}; // stages + 2 (u(0), u'(0), stage values)

        // p'(c_i) and p''(c_i) (derivatives in τ-space) as linear combinations of constraints:
        //   y = [u(0), dt*u'(0), U_0, ..., U_{s-1}]
        //   p'(c_i)  = d1_u0[i] * y0 + d1_dv0[i] * y1 + sum_j d1[i,j] * U_j
        //   p''(c_i) = d2_u0[i] * y0 + d2_dv0[i] * y1 + sum_j d2[i,j] * U_j
        std::vector<double> d1{}; // size stages*stages
        std::vector<double> d2{}; // size stages*stages
        std::vector<double> d1_u0{};  // size stages
        std::vector<double> d1_dv0{}; // size stages
        std::vector<double> d2_u0{};  // size stages
        std::vector<double> d2_dv0{}; // size stages

        // p(1), p'(1), p''(1) in τ-space as linear combinations of constraints y.
        std::vector<double> u1{};   // coefficients on U_j (size stages)
        std::vector<double> du1{};  // coefficients on U_j (size stages)
        std::vector<double> ddu1{}; // coefficients on U_j (size stages)
        double u1_u0{0.0};
        double u1_dv0{0.0};
        double du1_u0{0.0};
        double du1_dv0{0.0};
        double ddu1_u0{0.0};
        double ddu1_dv0{0.0};
    };

    auto buildSecondOrderCollocationData = [&](const CollocationMethod& method) -> SecondOrderCollocationData {
        SecondOrderCollocationData data;
        data.stages = method.stages;
        data.n_constraints = method.stages + 2;
        const int s = method.stages;
        const int n = data.n_constraints;

        std::vector<double> V(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        V[0] = 1.0;                         // u(0)
        V[static_cast<std::size_t>(n + 1)] = 1.0; // u'(0)

        for (int j = 0; j < s; ++j) {
            const double cj = method.c[static_cast<std::size_t>(j)];
            double pow = 1.0;
            const int row = 2 + j;
            for (int k = 0; k < n; ++k) {
                V[static_cast<std::size_t>(row * n + k)] = pow;
                pow *= cj;
            }
        }

        const auto Vinv = invertDenseMatrix(V, n);

        auto evalRow = [&](double tau, int deriv) -> std::vector<double> {
            std::vector<double> row(static_cast<std::size_t>(n), 0.0);
            if (deriv == 0) {
                double pow = 1.0;
                for (int k = 0; k < n; ++k) {
                    row[static_cast<std::size_t>(k)] = pow;
                    pow *= tau;
                }
                return row;
            }
            if (deriv == 1) {
                double pow = 1.0;
                for (int k = 1; k < n; ++k) {
                    row[static_cast<std::size_t>(k)] = static_cast<double>(k) * pow;
                    pow *= tau;
                }
                return row;
            }
            FE_THROW_IF(deriv != 2, InvalidArgumentException, "TimeLoop: invalid Hermite derivative order");
            double pow = 1.0;
            for (int k = 2; k < n; ++k) {
                row[static_cast<std::size_t>(k)] = static_cast<double>(k) * static_cast<double>(k - 1) * pow;
                pow *= tau;
            }
            return row;
        };

        auto applyMap = [&](std::span<const double> row) -> std::vector<double> {
            FE_CHECK_ARG(static_cast<int>(row.size()) == n, "TimeLoop: Hermite row size mismatch");
            std::vector<double> coeff(static_cast<std::size_t>(n), 0.0);
            for (int col = 0; col < n; ++col) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += row[static_cast<std::size_t>(k)] * Vinv[static_cast<std::size_t>(k * n + col)];
                }
                coeff[static_cast<std::size_t>(col)] = sum;
            }
            return coeff;
        };

        data.d1.resize(static_cast<std::size_t>(s) * static_cast<std::size_t>(s), 0.0);
        data.d2.resize(static_cast<std::size_t>(s) * static_cast<std::size_t>(s), 0.0);
        data.d1_u0.resize(static_cast<std::size_t>(s), 0.0);
        data.d1_dv0.resize(static_cast<std::size_t>(s), 0.0);
        data.d2_u0.resize(static_cast<std::size_t>(s), 0.0);
        data.d2_dv0.resize(static_cast<std::size_t>(s), 0.0);

        for (int i = 0; i < s; ++i) {
            const double ci = method.c[static_cast<std::size_t>(i)];
            const auto c1 = applyMap(evalRow(ci, 1));
            const auto c2 = applyMap(evalRow(ci, 2));

            data.d1_u0[static_cast<std::size_t>(i)] = c1[0];
            data.d1_dv0[static_cast<std::size_t>(i)] = c1[1];
            data.d2_u0[static_cast<std::size_t>(i)] = c2[0];
            data.d2_dv0[static_cast<std::size_t>(i)] = c2[1];

            for (int j = 0; j < s; ++j) {
                data.d1[static_cast<std::size_t>(i * s + j)] = c1[static_cast<std::size_t>(2 + j)];
                data.d2[static_cast<std::size_t>(i * s + j)] = c2[static_cast<std::size_t>(2 + j)];
            }
        }

        const auto u1c = applyMap(evalRow(1.0, 0));
        const auto du1c = applyMap(evalRow(1.0, 1));
        const auto ddu1c = applyMap(evalRow(1.0, 2));

        data.u1_u0 = u1c[0];
        data.u1_dv0 = u1c[1];
        data.du1_u0 = du1c[0];
        data.du1_dv0 = du1c[1];
        data.ddu1_u0 = ddu1c[0];
        data.ddu1_dv0 = ddu1c[1];

        data.u1.resize(static_cast<std::size_t>(s), 0.0);
        data.du1.resize(static_cast<std::size_t>(s), 0.0);
        data.ddu1.resize(static_cast<std::size_t>(s), 0.0);
        for (int j = 0; j < s; ++j) {
            data.u1[static_cast<std::size_t>(j)] = u1c[static_cast<std::size_t>(2 + j)];
            data.du1[static_cast<std::size_t>(j)] = du1c[static_cast<std::size_t>(2 + j)];
            data.ddu1[static_cast<std::size_t>(j)] = ddu1c[static_cast<std::size_t>(2 + j)];
        }

        return data;
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
        auto [ins_it, inserted] = cache.emplace(stages, buildSecondOrderCollocationData(method));
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

    auto ensureCollocationWorkspace = [&](int stages_needed) {
        if (collocation.jacobian && collocation.residual && collocation.delta &&
            collocation.stage_values && collocation.stage_combination && collocation.dv0 &&
            collocation.stages == stages_needed) {
            return;
        }

        collocation = {};
        collocation.stages = stages_needed;
        collocation.dt_integrator = std::make_shared<const Dt12NoHistoryIntegrator>();

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

        collocation.jacobian = factory.createMatrix(block_pattern);
        collocation.residual = factory.createVector(static_cast<GlobalIndex>(stages_needed) * n_dofs);
        collocation.delta = factory.createVector(static_cast<GlobalIndex>(stages_needed) * n_dofs);
        collocation.stage_values = factory.createVector(static_cast<GlobalIndex>(stages_needed) * n_dofs);
        collocation.stage_combination = factory.createVector(n_dofs);
        collocation.dv0 = factory.createVector(n_dofs);

        FE_CHECK_NOT_NULL(collocation.jacobian.get(), "TimeLoop: collocation jacobian");
        FE_CHECK_NOT_NULL(collocation.residual.get(), "TimeLoop: collocation residual");
        FE_CHECK_NOT_NULL(collocation.delta.get(), "TimeLoop: collocation delta");
        FE_CHECK_NOT_NULL(collocation.stage_values.get(), "TimeLoop: collocation stage_values");
        FE_CHECK_NOT_NULL(collocation.stage_combination.get(), "TimeLoop: collocation stage_combination");
        FE_CHECK_NOT_NULL(collocation.dv0.get(), "TimeLoop: collocation dv0");

        collocation.jacobian->zero();
        collocation.residual->zero();
        collocation.delta->zero();
        collocation.stage_values->zero();
        collocation.stage_combination->zero();
        collocation.dv0->zero();
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

        ensureCollocationWorkspace(method.stages);

        auto& sys = transient.system();
        const auto& constraints = sys.constraints();

        const CollocationFamily family = method.stiffly_accurate ? CollocationFamily::RadauIIA : CollocationFamily::Gauss;
        const SecondOrderCollocationData* so_data = nullptr;
        if (temporal_order == 2) {
            so_data = &getSecondOrderCollocationData(family, method.stages);
        }

        if (temporal_order == 2) {
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

    const double time_tol = 10.0 * std::numeric_limits<double>::epsilon()
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
        FE_THROW_IF(history.historyDepth() < vsvo_controller->maxOrder() + 1, InvalidArgumentException,
                    "TimeLoop: VSVO_BDF requires history depth >= max_order + 1");

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
                        history.ensureSecondOrderState(factory);

                        // If (u̇,ü) storage is missing (e.g., restart without second-order state),
                        // initialize from displacement history when possible.
                        if (!had_u_dot || !had_u_ddot) {
                            (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                                history,
                                history.uDot().localSpan(),
                                history.uDDot().localSpan(),
                                /*overwrite_u_dot=*/!had_u_dot,
                                /*overwrite_u_ddot=*/!had_u_ddot);
                        }

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
                        history.ensureSecondOrderState(factory);

                        if (!had_u_dot || !had_u_ddot) {
                            (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                                history,
                                history.uDot().localSpan(),
                                history.uDDot().localSpan(),
                                /*overwrite_u_dot=*/!had_u_dot,
                                /*overwrite_u_ddot=*/!had_u_ddot);
                        }

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

                    order = std::max(vsvo_controller->minOrder(), std::min(order, vsvo_controller->maxOrder()));
                    // Starter: restrict order to the amount of "real" history available.
                    order = std::max(1, std::min(order, history.stepIndex() + 1));
                    scheme_order = order;

                    const bool need_startup_reference = (history.stepIndex() == 0 && order == 1);
                    bool have_reference_solution = false;
                    if (need_startup_reference) {
                        // Bootstrap VSVO error estimation on the very first step. With no "real" history
                        // yet, extrapolation-based predictors reduce to u_pred = u^n and the correction
                        // u^{n+1} - u_pred measures solution change (O(dt)), not the local truncation
                        // error (O(dt^2) for BDF1). Use an embedded pair (BE vs CN) to obtain an
                        // O(dt^2) error estimate without requiring additional history.
                        NewtonReport nr_ref;
                        const int temporal_order = transient.system().temporalOrder();
                        if (temporal_order <= 1) {
                            nr_ref = solveThetaStep(/*theta=*/0.5, solve_time, dt);
                        } else if (temporal_order == 2) {
                            auto newmark_ref = std::make_shared<const NewmarkBetaIntegrator>(NewmarkBetaIntegratorOptions{
                                .beta = options_.newmark_beta,
                                .gamma = options_.newmark_gamma,
                            });

                            const bool had_u_dot = history.hasUDotState();
                            const bool had_u_ddot = history.hasUDDotState();
                            history.ensureSecondOrderState(factory);
                            if (!had_u_dot || !had_u_ddot) {
                                (void)utils::initializeSecondOrderStateFromDisplacementHistory(
                                    history,
                                    history.uDot().localSpan(),
                                    history.uDDot().localSpan(),
                                    /*overwrite_u_dot=*/!had_u_dot,
                                    /*overwrite_u_ddot=*/!had_u_ddot);
                            }

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
                        const double atol = vsvo_controller->absTol();
                        const double rtol = vsvo_controller->relTol();

                        const auto u = history.uSpan();
                        const auto ref = have_reference_solution ? scratch_vec1->localSpan() : vsvo_pred->localSpan();
                        FE_CHECK_ARG(u.size() == ref.size(), "TimeLoop: VSVO_BDF size mismatch");

                        double sum = 0.0;
                        for (std::size_t i = 0; i < u.size(); ++i) {
                            const double scale = atol + rtol * std::abs(static_cast<double>(u[i]));
                            const double e = static_cast<double>(u[i] - ref[i]);
                            sum += (e / scale) * (e / scale);
                        }
                        const double denom = (u.empty() ? 1.0 : static_cast<double>(u.size()));
                        error_norm = std::sqrt(sum / denom);
                    }
                    }
                } else if (options_.scheme == SchemeKind::TRBDF2) {
                    const double dt_saved = dt;
                    const double dt_prev_saved = history.dtPrev();
                    copyVector(*scratch_vec1, history.uPrev());
                    copyVector(*scratch_vec2, history.uPrev2());

                    bool restore_on_exit = true;
                    auto restore = [&]() {
                        history.setDt(dt_saved);
                        history.setPrevDt(dt_prev_saved);
                        copyVector(history.uPrev(), *scratch_vec1);
                        copyVector(history.uPrev2(), *scratch_vec2);
                        history.resetCurrentToPrevious();
                    };

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

                    if (restore_on_exit) {
                        restore();
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

                    const auto decision = options_.step_controller->onAccepted(info);
                    if (!decision.accept) {
                        if (callbacks.on_step_rejected) {
                            callbacks.on_step_rejected(history, StepRejectReason::ErrorTooLarge, nr);
                        }
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
                            FE_THROW_IF(!ga1_params.has_value(), systems::InvalidStateException,
                                        "TimeLoop: generalized-alpha parameters not initialized");

                            // Reconstruct u̇_n from displacement history using the same weights as the integrator.
                            const int available_history = history.historyDepth();
                            const int q_max = std::max(1, available_history - 1);
                            const int q = std::max(1, std::min(2, q_max));

                            const auto dt_hist = history.dtHistory();
                            const double dt_prev = (history.dtPrev() > 0.0 && std::isfinite(history.dtPrev()))
                                ? history.dtPrev()
                                : dt;

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

                            std::vector<double> nodes;
                            nodes.reserve(static_cast<std::size_t>(q + 1));
                            nodes.push_back(0.0);
                            double accum = 0.0;
                            for (int j = 1; j <= q; ++j) {
                                accum += historyDt(j - 1);
                                nodes.push_back(-accum);
                            }

                            const auto w = math::finiteDifferenceWeights(/*derivative_order=*/1, /*x0=*/0.0, nodes);
                            FE_THROW_IF(static_cast<int>(w.size()) != q + 1, systems::InvalidStateException,
                                        "TimeLoop: generalized-alpha history weight mismatch");

                            // u̇_n
                            std::fill(dt_u_dot.begin(), dt_u_dot.end(), static_cast<Real>(0.0));
                            for (int j = 0; j <= q; ++j) {
                                const auto uj = history.uPrevKSpan(j + 1);
                                FE_CHECK_ARG(uj.size() == dt_u_dot.size(), "TimeLoop: generalized-alpha history size mismatch");
                                const double alpha = w[static_cast<std::size_t>(j)];
                                for (std::size_t k = 0; k < dt_u_dot.size(); ++k) {
                                    dt_u_dot[k] += static_cast<Real>(alpha) * uj[k];
                                }
                            }

                            // dt*u̇_{n+1} = (u_{n+1} - u_n)/gamma - dt*((1-gamma)/gamma)*u̇_n
                            const double gamma = ga1_params->gamma;
                            const double inv_gamma = 1.0 / gamma;
                            const double c_hist = -dt * (1.0 - gamma) * inv_gamma;
                            const auto u_n = history.uPrevSpan();
                            FE_CHECK_ARG(u_n.size() == u_np1.size(), "TimeLoop: generalized-alpha u_n size mismatch");
                            for (std::size_t k = 0; k < dt_u_dot.size(); ++k) {
                                dt_u_dot[k] = static_cast<Real>(inv_gamma) * (u_np1[k] - u_n[k]) +
                                    static_cast<Real>(c_hist) * dt_u_dot[k];
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

    report.success = false;
    report.steps_taken = options_.max_steps;
    report.final_time = history.time();
    report.message = "TimeLoop: max_steps exceeded";
    return report;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
