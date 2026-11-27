/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Core/Types.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FieldBinding.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/GeneratedState.h"
#include "TimeStepping/StepController.h"
#include "TimeStepping/VSVO_BDF_Controller.h"
#include "TimeStepping/NewmarkBeta.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace ts_test = svmp::FE::timestepping::test;

namespace {

struct ContextSnapshot {
    double time{0.0};
    double dt{0.0};
    double time_weight{1.0};
    double non_time_weight{1.0};
    std::vector<double> dt1{};
};

bool sameSnapshot(const ContextSnapshot& a, const ContextSnapshot& b, double tol)
{
    if (std::abs(a.time - b.time) > tol) return false;
    if (std::abs(a.dt - b.dt) > tol) return false;
    if (std::abs(a.time_weight - b.time_weight) > tol) return false;
    if (std::abs(a.non_time_weight - b.non_time_weight) > tol) return false;
    if (a.dt1.size() != b.dt1.size()) return false;
    for (std::size_t i = 0; i < a.dt1.size(); ++i) {
        if (std::abs(a.dt1[i] - b.dt1[i]) > 1e-12) return false;
    }
    return true;
}

struct RecordedContexts {
    std::vector<ContextSnapshot> snapshots{};
};

class RecordingKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    RecordingKernel(std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner, RecordedContexts* out)
        : inner_(std::move(inner))
        , out_(out)
    {
        if (!inner_) {
            throw std::runtime_error("RecordingKernel: inner is null");
        }
        if (!out_) {
            throw std::runtime_error("RecordingKernel: out is null");
        }
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        return inner_->getRequiredData();
    }

    [[nodiscard]] svmp::FE::assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return inner_->materialStateSpec();
    }

    [[nodiscard]] std::vector<svmp::FE::params::Spec> parameterSpecs() const override
    {
        return inner_->parameterSpecs();
    }

    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override
    {
        return inner_->maxTemporalDerivativeOrder();
    }

    [[nodiscard]] bool hasCell() const noexcept override { return inner_->hasCell(); }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return inner_->hasBoundaryFace(); }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return inner_->hasInteriorFace(); }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        inner_->computeCell(ctx, output);

        const auto* ti = ctx.timeIntegrationContext();
        if (!ti) {
            return;
        }

        ContextSnapshot snap;
        snap.time = static_cast<double>(ctx.time());
        snap.dt = static_cast<double>(ctx.timeStep());
        snap.time_weight = static_cast<double>(ti->time_derivative_term_weight);
        snap.non_time_weight = static_cast<double>(ti->non_time_derivative_term_weight);
        if (ti->dt1) {
            snap.dt1.reserve(ti->dt1->a.size());
            for (const auto coeff : ti->dt1->a) {
                snap.dt1.push_back(static_cast<double>(coeff));
            }
        }

        constexpr double tol = 1e-15;
        for (const auto& existing : out_->snapshots) {
            if (sameSnapshot(existing, snap, tol)) {
                return;
            }
        }
        out_->snapshots.push_back(std::move(snap));
    }

    void computeBoundaryFace(const svmp::FE::assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             svmp::FE::assembly::KernelOutput& output) override
    {
        inner_->computeBoundaryFace(ctx, boundary_marker, output);
    }

    void computeInteriorFace(const svmp::FE::assembly::AssemblyContext& ctx_minus,
                             const svmp::FE::assembly::AssemblyContext& ctx_plus,
                             svmp::FE::assembly::KernelOutput& output_minus,
                             svmp::FE::assembly::KernelOutput& output_plus,
                             svmp::FE::assembly::KernelOutput& coupling_minus_plus,
                             svmp::FE::assembly::KernelOutput& coupling_plus_minus) override
    {
        inner_->computeInteriorFace(ctx_minus, ctx_plus, output_minus, output_plus, coupling_minus_plus, coupling_plus_minus);
    }

    [[nodiscard]] std::string name() const override
    {
        return "Recording(" + inner_->name() + ")";
    }

private:
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner_{};
    RecordedContexts* out_{nullptr};
};

struct RunResult {
    RecordedContexts ctxs;
    std::vector<Real> final_u;
    std::vector<double> accepted_steps;
};

RunResult runOneStep(svmp::FE::timestepping::SchemeKind scheme,
                     double dt,
                     double lambda,
                     std::function<void(svmp::FE::timestepping::TimeLoopOptions&)> configure_opts = {},
                     std::optional<double> previous_step = std::nullopt)
{
    RunResult out;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto inner_kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    auto kernel = std::make_shared<RecordingKernel>(inner_kernel, &out.ctxs);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);
    EXPECT_TRUE(sys.isSetup());

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return out;
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    EXPECT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/3);
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    for (int k = 1; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), u0);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(previous_step.value_or(dt));

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = dt;
    opts.dt = dt;
    opts.max_steps = 10;
    opts.scheme = scheme;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 10;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    if (configure_opts) {
        configure_opts(opts);
    }

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history);
    EXPECT_TRUE(rep.success) << rep.message;
    EXPECT_NEAR(rep.final_time, dt, 1e-12);

    out.final_u = ts_test::getVectorByDof(history.uPrev());
    out.accepted_steps.assign(history.dtHistory().begin(), history.dtHistory().end());
    return out;
}

const ContextSnapshot* findSnapshot(const RecordedContexts& ctxs,
                                    double time,
                                    double time_weight,
                                    double non_time_weight,
                                    double tol = 1e-14)
{
    for (const auto& s : ctxs.snapshots) {
        if (std::abs(s.time - time) > tol) continue;
        if (std::abs(s.time_weight - time_weight) > tol) continue;
        if (std::abs(s.non_time_weight - non_time_weight) > tol) continue;
        return &s;
    }
    return nullptr;
}

} // namespace

TEST(TimeLoopStageTimes, GeneralizedAlphaUsesAlphaFStageTime)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double dt = 0.1;
    const double lambda = 1.0;

    auto res = runOneStep(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                          dt,
                          lambda,
                          [](svmp::FE::timestepping::TimeLoopOptions& o) {
                              // rho_inf=1 => alpha_f=1/(1+rho_inf)=0.5 (1st-order generalized-alpha)
                              o.generalized_alpha_rho_inf = 1.0;
                          });

    ASSERT_FALSE(res.ctxs.snapshots.empty());
    const double expected_stage_time = 0.5 * dt;

    const bool saw_stage_time = std::any_of(res.ctxs.snapshots.begin(),
                                            res.ctxs.snapshots.end(),
                                            [&](const ContextSnapshot& s) {
                                                return std::abs(s.time - expected_stage_time) <= 1e-14;
                                            });
    EXPECT_TRUE(saw_stage_time);
}

TEST(TimeLoopStageTimes, TRBDF2UsesExpectedStageTimesAndWeights)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double dt = 0.2;
    const double gamma = 0.25;
    const double lambda = 1.0;
    const double dt1 = gamma * dt;
    const double dt2 = dt - dt1;
    const double previous_step = 0.3;

    auto res = runOneStep(svmp::FE::timestepping::SchemeKind::TRBDF2,
                          dt,
                          lambda,
                          [&](svmp::FE::timestepping::TimeLoopOptions& o) {
                              o.trbdf2_gamma = gamma;
                          }, previous_step);

    ASSERT_FALSE(res.ctxs.snapshots.empty());
    ASSERT_GE(res.accepted_steps.size(), 2u);
    EXPECT_DOUBLE_EQ(res.accepted_steps[0], dt);
    EXPECT_DOUBLE_EQ(res.accepted_steps[1], previous_step);

    // Stage 1 TR split: current non-time term weight is 0.5 at t + dt1.
    const auto* stage1 = findSnapshot(res.ctxs, /*time=*/dt1, /*time_weight=*/1.0, /*non_time_weight=*/0.5);
    ASSERT_NE(stage1, nullptr);
    ASSERT_EQ(stage1->dt1.size(), 2u);
    EXPECT_NEAR(stage1->dt, dt1, 1e-14);
    EXPECT_NEAR(stage1->dt1[0], 1.0 / dt1, 1e-14);
    EXPECT_NEAR(stage1->dt1[1], -1.0 / dt1, 1e-14);

    // Stage 2 BDF2 step over dt2 on [t+dt1, t+dt]: dt1 weights should match the variable-step BDF2 stencil.
    const auto* stage2 = findSnapshot(res.ctxs, /*time=*/dt, /*time_weight=*/1.0, /*non_time_weight=*/1.0);
    ASSERT_NE(stage2, nullptr);
    ASSERT_EQ(stage2->dt1.size(), 3u);
    EXPECT_NEAR(stage2->dt, dt2, 1e-14);

    const double r = dt2 / dt1;
    const double inv_dt2 = 1.0 / dt2;
    EXPECT_NEAR(stage2->dt1[0], ((1.0 + 2.0 * r) / (1.0 + r)) * inv_dt2, 1e-12);
    EXPECT_NEAR(stage2->dt1[1], (-(1.0 + r)) * inv_dt2, 1e-12);
    EXPECT_NEAR(stage2->dt1[2], ((r * r) / (1.0 + r)) * inv_dt2, 1e-12);
}

// The time semantics of the stage solve of each scheme: the stage time t(),
// the physical step deltat() and deltat_eff(), which the assembly kernels
// and the JIT evaluate as 1/|a0| of the dt(.) stencil.
TEST(TimeLoopStageTimes, StageTimeStepAndEffectiveStepPerScheme)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using svmp::FE::timestepping::SchemeKind;
    using svmp::FE::timestepping::TimeLoopOptions;
    const double dt = 0.1;
    // Generalized-alpha (first order): alpha_m = (3 - rho)/(2 (1 + rho)),
    // alpha_f = 1/(1 + rho), gamma = 1/2 + alpha_m - alpha_f.
    const auto alpha = [](double rho_inf) {
        const double alpha_m = (3.0 - rho_inf) / (2.0 * (1.0 + rho_inf));
        const double alpha_f = 1.0 / (1.0 + rho_inf);
        const double gamma = 0.5 + alpha_m - alpha_f;
        return std::array<double, 3>{alpha_m, alpha_f, gamma};
    };
    const auto ga05 = alpha(0.5);
    const auto ga10 = alpha(1.0);

    struct Row {
        const char* name;
        SchemeKind scheme;
        std::function<void(TimeLoopOptions&)> configure;
        double stage_time;
        double effective_dt;
    };
    const std::vector<Row> table = {
        {"backward Euler", SchemeKind::BackwardEuler, {}, dt, dt},
        // The first BDF2 step is a backward Euler step (the start-up).
        {"BDF2, first step", SchemeKind::BDF2, {}, dt, dt},
        {"theta = 0.5", SchemeKind::ThetaMethod, [](TimeLoopOptions& o) { o.theta = 0.5; }, dt, dt},
        {"generalized-alpha rho_inf = 0.5", SchemeKind::GeneralizedAlpha,
         [](TimeLoopOptions& o) { o.generalized_alpha_rho_inf = 0.5; },
         ga05[1] * dt, ga05[2] * ga05[1] * dt / ga05[0]},
        {"generalized-alpha rho_inf = 1", SchemeKind::GeneralizedAlpha,
         [](TimeLoopOptions& o) { o.generalized_alpha_rho_inf = 1.0; },
         ga10[1] * dt, ga10[2] * ga10[1] * dt / ga10[0]},
    };
    EXPECT_NEAR(table[3].effective_dt, 0.5333333333333333 * dt, 1e-15);

    for (const auto& row : table) {
        SCOPED_TRACE(row.name);
        const auto res = runOneStep(row.scheme, dt, /*lambda=*/1.0, row.configure);
        const auto stage = std::find_if(res.ctxs.snapshots.begin(), res.ctxs.snapshots.end(),
                                        [&](const ContextSnapshot& s) {
                                            return std::abs(s.time - row.stage_time) <= 1e-14 && !s.dt1.empty();
                                        });
        ASSERT_NE(stage, res.ctxs.snapshots.end());
        EXPECT_NEAR(stage->dt, dt, 1e-15);
        EXPECT_NEAR(1.0 / std::abs(stage->dt1[0]), row.effective_dt, 1e-14);
    }
}

// previousSolution(k) reads history slot k.  The first-order generalized-alpha
// stage solve keeps the injected rate u_dot^n in slot 2, and the second-order
// generalized-alpha and Newmark solves keep their scheme constants in slots 1
// and 2.  A form that would read such a slot is refused at assembly, with the
// integrator named; the other slots are accepted.
TEST(TimeLoopStageTimes, PreviousSolutionOfAnInjectedHistorySlotIsRefused)
{
    namespace forms = svmp::FE::forms;
    namespace systems = svmp::FE::systems;
    namespace ts = svmp::FE::timestepping;
    // The message of the refusal, empty when the residual assembles.
    const auto assembleWith = [](int history_slot, int dt_order,
                                 std::shared_ptr<const systems::TimeIntegrator> integrator) {
        auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
        auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
        systems::FESystem sys(mesh);
        const auto u_id = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("op");
        const auto u = forms::FormExpr::stateField(u_id, *space, "u");
        const auto v = forms::FormExpr::testFunction(u_id, *space, "v");
        const auto residual =
            ((forms::dt(u, dt_order) + u - forms::FormExpr::previousSolution(history_slot)) * v).dx();
        (void)systems::installFormulation(sys, "op", {u_id}, residual);
        systems::SetupInputs inputs;
        inputs.topology_override = ts_test::singleTetraTopology();
        sys.setup({}, inputs);

        const auto n = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
        const std::vector<Real> values(n, 1.0), previous(n, 0.5), rate(n, 0.25), older(n, 0.125);
        const std::array<std::span<const Real>, 3> history{previous, rate, older};
        systems::SystemStateView state;
        state.dt = 0.1;
        state.dt_prev = 0.1;
        state.u = values;
        state.u_prev = previous;
        state.u_prev2 = rate;
        state.u_history = history;
        systems::TransientSystem transient(sys, std::move(integrator));
        systems::AssemblyRequest req;
        req.op = "op";
        req.want_vector = true;
        svmp::FE::assembly::DenseVectorView R(static_cast<GlobalIndex>(n));
        try {
            (void)transient.assemble(req, state, nullptr, &R);
        } catch (const std::exception& error) {
            return std::string(error.what());
        }
        return std::string{};
    };

    const auto generalized_alpha = std::make_shared<ts::GeneralizedAlphaFirstOrderIntegrator>(
        ts::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = 5.0 / 6.0, .alpha_f = 2.0 / 3.0, .gamma = 2.0 / 3.0, .history_rate_order = 0});
    EXPECT_EQ(assembleWith(1, 1, generalized_alpha), "");
    const auto refused = assembleWith(2, 1, generalized_alpha);
    EXPECT_NE(refused.find("previousSolution(2)"), std::string::npos) << refused;
    EXPECT_NE(refused.find("GeneralizedAlpha(1stOrder)"), std::string::npos) << refused;

    EXPECT_EQ(assembleWith(2, 1, std::make_shared<systems::BackwardDifferenceIntegrator>()), "");

    const auto newmark = std::make_shared<ts::NewmarkBetaIntegrator>(ts::NewmarkBetaIntegratorOptions{});
    const auto refused_newmark = assembleWith(1, 2, newmark);
    EXPECT_NE(refused_newmark.find("NewmarkBeta"), std::string::npos) << refused_newmark;
    const auto generalized_alpha_2 = std::make_shared<ts::GeneralizedAlphaSecondOrderIntegrator>(
        ts::GeneralizedAlphaSecondOrderIntegratorOptions{.alpha_m = 0.5, .alpha_f = 0.5, .beta = 0.25, .gamma = 0.5});
    const auto refused_alpha_2 = assembleWith(2, 2, generalized_alpha_2);
    EXPECT_NE(refused_alpha_2.find("GeneralizedAlpha(2ndOrder)"), std::string::npos) << refused_alpha_2;
}

// historyWeightedSum and historyConvolution read slot k for their k-th weight.
// Under first-order generalized-alpha the weights (1, 1) summed the previous
// solution 0.5 and the injected rate instead of two previous solutions.
TEST(TimeLoopStageTimes, WeightedHistoryOfAnInjectedHistorySlotIsRefused)
{
    namespace forms = svmp::FE::forms;
    namespace systems = svmp::FE::systems;
    namespace ts = svmp::FE::timestepping;
    // The message of the refusal, empty when the residual assembles.
    const auto assembleWith = [](const std::function<forms::FormExpr(std::vector<forms::FormExpr>)>& history_sum,
                                 std::size_t n_weights,
                                 std::shared_ptr<const systems::TimeIntegrator> integrator) {
        auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
        auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
        systems::FESystem sys(mesh);
        const auto u_id = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("op");
        const auto u = forms::FormExpr::stateField(u_id, *space, "u");
        const auto v = forms::FormExpr::testFunction(u_id, *space, "v");
        const std::vector<forms::FormExpr> weights(n_weights, forms::FormExpr::constant(1.0));
        const auto residual = ((forms::dt(u) + u - history_sum(weights)) * v).dx();
        (void)systems::installFormulation(sys, "op", {u_id}, residual);
        systems::SetupInputs inputs;
        inputs.topology_override = ts_test::singleTetraTopology();
        sys.setup({}, inputs);

        const auto n = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
        const std::vector<Real> values(n, 1.0), previous(n, 0.5), rate(n, 0.25), older(n, 0.125);
        const std::array<std::span<const Real>, 3> history{previous, rate, older};
        systems::SystemStateView state;
        state.dt = 0.1;
        state.dt_prev = 0.1;
        state.u = values;
        state.u_prev = previous;
        state.u_prev2 = rate;
        state.u_history = history;
        systems::TransientSystem transient(sys, std::move(integrator));
        systems::AssemblyRequest req;
        req.op = "op";
        req.want_vector = true;
        svmp::FE::assembly::DenseVectorView R(static_cast<GlobalIndex>(n));
        try {
            (void)transient.assemble(req, state, nullptr, &R);
        } catch (const std::exception& error) {
            return std::string(error.what());
        }
        return std::string{};
    };

    const auto generalized_alpha = std::make_shared<ts::GeneralizedAlphaFirstOrderIntegrator>(
        ts::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = 5.0 / 6.0, .alpha_f = 2.0 / 3.0, .gamma = 2.0 / 3.0, .history_rate_order = 0});
    const auto bdf = std::make_shared<systems::BackwardDifferenceIntegrator>();
    for (const auto& sum : {std::function<forms::FormExpr(std::vector<forms::FormExpr>)>(
                                &forms::FormExpr::historyWeightedSum),
                            std::function<forms::FormExpr(std::vector<forms::FormExpr>)>(
                                &forms::FormExpr::historyConvolution)}) {
        EXPECT_EQ(assembleWith(sum, 1, generalized_alpha), "");
        const auto refused = assembleWith(sum, 2, generalized_alpha);
        EXPECT_NE(refused.find("history slot 2"), std::string::npos) << refused;
        EXPECT_NE(refused.find("GeneralizedAlpha(1stOrder)"), std::string::npos) << refused;
        EXPECT_EQ(assembleWith(sum, 2, bdf), "");
        // Weights supplied at assembly can reach every stored slot.
        const auto refused_runtime = assembleWith(sum, 0, generalized_alpha);
        EXPECT_NE(refused_runtime.find("assembly-time weights"), std::string::npos) << refused_runtime;
    }
}

// ---------------------------------------------------------------------------
// The generated-state schedule compiled from declarations drives TimeLoop as
// the transcribed selection does (design note 12, level 3): backward Euler
// and generalized-alpha (whose endpoint candidate is regenerated after the
// stage solve), accepted steps, and attempts the outer fixed point does not
// converge, which the step controller rejects and retries from the restored
// time step.  The traces of the synchronization points, the reports and the
// final states are compared.
// ---------------------------------------------------------------------------

namespace {

namespace gs = svmp::FE::timestepping;

struct ScheduledLoopRun {
    std::vector<std::string> trace{};
    bool success{false};
    int steps{0};
    std::vector<Real> u{};
};

ScheduledLoopRun runScheduledLoop(bool from_plan,
                                  gs::SchemeKind scheme,
                                  const gs::OuterFixedPointOverrides& overrides)
{
    ScheduledLoopRun run;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    // A generated coefficient m = 1 + 0.9 * u(vertex 0), regenerated at each
    // synchronization of the geometry role.
    auto generated = std::make_shared<double>(1.0);
    const auto m = svmp::FE::forms::FormExpr::coefficient(
        "generated_m", [generated](Real, Real, Real) { return static_cast<Real>(*generated); });
    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + m * u * v - v).dx();
    svmp::FE::forms::FormCompiler compiler;
    sys.addCellKernel("op", u_field, u_field,
                      std::make_shared<svmp::FE::forms::NonlinearFormKernel>(compiler.compileResidual(form),
                                                                            svmp::FE::forms::ADMode::Forward));
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);
    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);
    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available";
        return run;
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = gs::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/3);
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    for (int k = 1; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), u0);
    }
    history.resetCurrentToPrevious();
    const double dt = 0.1;
    history.setPrevDt(dt);

    gs::GeneratedStatePolicy policy;
    gs::NewtonOptions::ExternalStateFixedPointOptions fixed_point;
    if (from_plan) {
        gs::GeneratedStateDescriptor geometry;
        geometry.enablement = gs::OuterEnablement::RequiredWhenPresent;
        geometry.reads = {{.what = {.kind = gs::ResourceKind::Field, .field = u_field},
                           .version = gs::StateVersion::CurrentTrial}};
        for (const auto phase : {gs::WritePhase::ConstraintConstruction, gs::WritePhase::ProjectedRefresh}) {
            geometry.writes.push_back({.what = {.kind = gs::ResourceKind::AffineConstraints},
                                       .phase = phase,
                                       .residual_defining = true,
                                       .requires_fresh_residual = true});
        }
        const auto plan = gs::compileGeneratedStateSchedule(
            {.solve = gs::GeneratedStateSolve::Transient,
             .purpose = gs::GeneratedStatePurpose::TransientAttempt,
             .block_fields = {u_field},
             .overrides = overrides},
            {{.name = "geometry", .role = gs::GeneratedStateRole::Geometry, .descriptor = geometry}}, {});
        policy = plan.selection.policy;
        fixed_point = plan.fixed_point;
    } else {
        const auto selection = ts_test::transcribedGeneratedStateSelection(
            {.cut_geometry = true}, gs::GeneratedStateSolve::Transient, /*per_step_only=*/false);
        policy = selection.policy;
        fixed_point = gs::outerFixedPointPreset(gs::GeneratedStateSolve::Transient, overrides);
        fixed_point.enabled = selection.policy.outer_fixed_point;
    }

    gs::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 3.0 * dt;
    opts.dt = dt;
    opts.max_steps = 20;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    opts.newton.external_state_fixed_point = fixed_point;
    opts.step_controller = std::make_shared<gs::SimpleStepController>(
        gs::SimpleStepControllerOptions{.max_retries = 2});
    auto registry = std::make_shared<gs::GeneratedStateRegistry>(policy);
    registry->add(std::make_shared<gs::FunctionGeneratedStateParticipant>(
        "geometry", gs::GeneratedStateRole::Geometry,
        gs::FunctionGeneratedStateParticipant::Callbacks{
            .synchronize =
                [&run, generated](const svmp::FE::systems::SystemStateView& state, gs::GeneratedStateSync& sync) {
                    const double u_vertex = state.u.empty() ? 0.0 : static_cast<double>(state.u.front());
                    *generated = 1.0 + 0.9 * u_vertex;
                    run.trace.push_back(std::string(gs::generatedStateSyncPointName(sync.point)) + "@" +
                                        std::to_string(state.time) + ":" + std::to_string(u_vertex));
                }}));
    registry->bind(opts.newton);

    gs::TimeLoop loop(opts);
    const auto report = loop.run(transient, *factory, *linear, history);
    run.success = report.success;
    run.steps = report.steps_taken;
    run.u = ts_test::getVectorByDof(history.uPrev());
    return run;
}

} // namespace

TEST(GeneratedStateScheduleTimeLoop, BackwardEulerAndGeneralizedAlphaRunAlikeUnderBothSchedules)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    for (const auto scheme : {gs::SchemeKind::BackwardEuler, gs::SchemeKind::GeneralizedAlpha}) {
        for (const auto& overrides : {gs::OuterFixedPointOverrides{},
                                      // Two passes cannot converge the outer
                                      // fixed point: every attempt is rejected
                                      // and retried from the restored step.
                                      gs::OuterFixedPointOverrides{.max_passes = 2}}) {
            const std::string where = std::string(scheme == gs::SchemeKind::BackwardEuler ? "BE" : "GA") +
                                      " max_passes=" + std::to_string(overrides.max_passes.value_or(30));
            const auto legacy = runScheduledLoop(false, scheme, overrides);
            const auto plan = runScheduledLoop(true, scheme, overrides);
            EXPECT_FALSE(legacy.trace.empty()) << where;
            EXPECT_EQ(legacy.trace, plan.trace) << where;
            EXPECT_EQ(legacy.success, plan.success) << where;
            EXPECT_EQ(legacy.steps, plan.steps) << where;
            EXPECT_EQ(legacy.u, plan.u) << where;
            EXPECT_EQ(legacy.success, !overrides.max_passes.has_value()) << where;
            const auto seen = [&](const std::string& token) {
                return std::any_of(legacy.trace.begin(), legacy.trace.end(), [&](const std::string& entry) {
                    return entry.rfind(token + "@", 0) == 0;
                });
            };
            // The generalized-alpha endpoint is regenerated after a stage solve
            // that converged; backward Euler has no endpoint points.
            EXPECT_EQ(seen("endpoint_candidate"),
                      scheme == gs::SchemeKind::GeneralizedAlpha && legacy.success)
                << where;
            // A rejected attempt restores the accepted time step.
            EXPECT_EQ(seen("restored_time_step"), overrides.max_passes.has_value()) << where;
        }
    }
}

// ---------------------------------------------------------------------------
// previous(u, k) holds u at step n - k on every scheme (decision D66 item 2).
// The shadow participant reads the accepted history at each attempt entry and
// keeps it through the attempt, where first-order generalized-alpha and the
// second-order schemes hold rates in history slots during the stage solve.
// Each case runs three steps with the shadows refreshed at the attempt entry,
// at every projected outer refresh and at the endpoint; one attempt is
// rejected and retried, and a second loop continues from the history.  The
// residual reads both shadows, so the explicit evaluation of the BDF2 starter
// (a theta = 0.5 step, which refreshes no generated state) reads the values
// of its attempt entry.
// ---------------------------------------------------------------------------

namespace {

struct PreviousShadowCase {
    std::string name;
    gs::SchemeKind scheme;
    int order{1};
    double rho_inf{1.0};
};

// Every value the two shadows took, with the accepted solutions they must equal.
struct PreviousShadowRun {
    int mismatches{0};
    int refreshes{0};
    int attempts{0};
    bool success{false};
    std::string first_mismatch{};
};

PreviousShadowRun runPreviousShadows(const PreviousShadowCase& item)
{
    PreviousShadowRun run;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    svmp::FE::systems::FieldBinding fields(sys, "previous shadow test");
    const auto U = fields.unknownSymbols(svmp::FE::systems::FieldSpec{.name = "u", .space = space}, "u", "u", "v");
    const auto previous1 = fields.previous(U, 1);
    const auto previous2 = fields.previous(U, 2);
    // The residual reads both shadows, so every evaluation needs them defined,
    // the explicit part of the BDF2 starter (a theta = 0.5 step) included.
    const auto residual =
        ((svmp::FE::forms::dt(U.state, item.order) + U.state +
          svmp::FE::forms::FormExpr::constant(0.1) * (previous1 - previous2)) *
         U.test)
            .dx();
    (void)svmp::FE::systems::installFormulation(sys, "op", {U.id}, residual);
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);
    const auto shadows = svmp::FE::systems::shadowFields(sys);
    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);
    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available";
        return run;
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    const auto n_dofs = sys.dofHandler().getNumDofs();
    const double dt = 0.1;
    auto history = item.order == 1
        ? gs::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/3)
        : gs::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/2, /*allocate_second_order_state=*/true);
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    std::vector<Real> u_older(u0.size());
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_older[i] = static_cast<Real>(0.9 * static_cast<double>(u0[i]));
    }
    ts_test::setVectorByDof(history.uPrev(), u0);
    for (int k = 2; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), item.order == 1 ? u0 : u_older);
    }
    if (item.order == 2) {
        ts_test::setVectorByDof(history.uDot(), std::vector<Real>(u0.size(), 0.0));
        std::vector<Real> a0(u0.size());
        for (std::size_t i = 0; i < u0.size(); ++i) {
            a0[i] = -u0[i];
        }
        ts_test::setVectorByDof(history.uDDot(), a0);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    // The accepted solutions, oldest first: the initial history, then each
    // accepted step.
    std::vector<std::vector<Real>> accepted{item.order == 1 ? u0 : u_older, u0};
    std::vector<Real> expected1, expected2;
    const auto check = [&](const char* where) {
        ++run.refreshes;
        const auto previous1 = sys.prescribedFieldCoefficients(shadows[0].shadow);
        const auto previous2 = sys.prescribedFieldCoefficients(shadows[1].shadow);
        const bool same = std::equal(previous1.begin(), previous1.end(), expected1.begin(), expected1.end()) &&
                          std::equal(previous2.begin(), previous2.end(), expected2.begin(), expected2.end());
        if (!same) {
            if (run.mismatches == 0) {
                run.first_mismatch = std::string(where) + " attempt " + std::to_string(run.attempts);
            }
            ++run.mismatches;
        }
    };

    auto shadow_participant = gs::makeShadowFieldParticipant(sys);
    auto registry = std::make_shared<gs::GeneratedStateRegistry>(
        gs::GeneratedStatePolicy{.solve = gs::GeneratedStateSolve::Transient, .outer_fixed_point = true});
    registry->add(std::make_shared<gs::FunctionGeneratedStateParticipant>(
        "geometry", gs::GeneratedStateRole::Geometry, gs::FunctionGeneratedStateParticipant::Callbacks{}));
    registry->add(shadow_participant);
    registry->add(std::make_shared<gs::FunctionGeneratedStateParticipant>(
        "recorder", gs::GeneratedStateRole::Reference,
        gs::FunctionGeneratedStateParticipant::Callbacks{
            .synchronize = [&](const svmp::FE::systems::SystemStateView&, gs::GeneratedStateSync& sync) {
                check(gs::generatedStateSyncPointName(sync.point));
            }}));

    bool rejected = false;
    gs::TimeLoopCallbacks callbacks;
    callbacks.on_step_start = [&](const gs::TimeHistory&) {
        ++run.attempts;
        expected1 = accepted[accepted.size() - 1];
        expected2 = accepted[accepted.size() - 2];
    };
    callbacks.on_before_step_accept = [&](gs::TimeHistory&, const gs::NewtonReport&) {
        // Reject the first attempt of the second step once.
        if (!rejected && accepted.size() == 3) {
            rejected = true;
            return false;
        }
        return true;
    };
    callbacks.on_step_accepted = [&](gs::TimeHistory& h) {
        accepted.push_back(ts_test::getVectorByDof(h.uPrev()));
    };

    const auto loop_options = [&](double t0) {
        gs::TimeLoopOptions opts;
        opts.t0 = t0;
        opts.t_end = t0 + 3.0 * dt;
        opts.dt = dt;
        opts.max_steps = 20;
        opts.scheme = item.scheme;
        opts.generalized_alpha_rho_inf = item.rho_inf;
        opts.newton.residual_op = "op";
        opts.newton.jacobian_op = "op";
        opts.newton.max_iterations = 8;
        opts.newton.abs_tolerance = 1e-12;
        opts.newton.rel_tolerance = 0.0;
        opts.newton.external_state_fixed_point =
            gs::outerFixedPointPreset(gs::GeneratedStateSolve::Transient);
        opts.newton.external_state_fixed_point.enabled = true;
        if (item.scheme == gs::SchemeKind::VSVO_BDF) {
            // Error control loose enough that only the forced rejection
            // rejects an attempt.
            gs::VSVO_BDF_ControllerOptions controller;
            controller.abs_tol = 1.0e3;
            controller.rel_tol = 1.0e3;
            controller.max_order = 2;
            controller.max_retries = 2;
            opts.step_controller = std::make_shared<gs::VSVO_BDF_Controller>(controller);
        } else {
            opts.step_controller = std::make_shared<gs::SimpleStepController>(
                gs::SimpleStepControllerOptions{.max_retries = 2, .increase_factor = 1.0});
        }
        registry->bind(opts.newton);
        return opts;
    };
    gs::TimeLoop first(loop_options(0.0));
    const auto first_report = first.run(transient, *factory, *linear, history, callbacks);
    // A second loop continues from the history (a restart of the run).
    gs::TimeLoop second(loop_options(history.time()));
    const auto second_report = second.run(transient, *factory, *linear, history, callbacks);
    run.success = first_report.success && second_report.success && rejected;
    return run;
}

} // namespace

TEST(GeneratedStatePreviousShadows, HoldTheAcceptedSolutionsOnEveryScheme)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const std::vector<PreviousShadowCase> cases{
        {"backward Euler", gs::SchemeKind::BackwardEuler},
        {"BDF2", gs::SchemeKind::BDF2},
        {"theta 0.5", gs::SchemeKind::ThetaMethod},
        {"TR-BDF2", gs::SchemeKind::TRBDF2},
        {"VSVO-BDF", gs::SchemeKind::VSVO_BDF},
        {"generalized-alpha first order, rho_inf 0.5", gs::SchemeKind::GeneralizedAlpha, 1, 0.5},
        {"generalized-alpha first order, rho_inf 1", gs::SchemeKind::GeneralizedAlpha, 1, 1.0},
        {"generalized-alpha second order, rho_inf 0.5", gs::SchemeKind::GeneralizedAlpha, 2, 0.5},
        {"Newmark", gs::SchemeKind::Newmark, 2},
    };
    for (const auto& item : cases) {
        const auto run = runPreviousShadows(item);
        EXPECT_TRUE(run.success) << item.name;
        EXPECT_GT(run.refreshes, run.attempts) << item.name;
        EXPECT_EQ(run.mismatches, 0) << item.name << ": first at " << run.first_mismatch;
    }
}
