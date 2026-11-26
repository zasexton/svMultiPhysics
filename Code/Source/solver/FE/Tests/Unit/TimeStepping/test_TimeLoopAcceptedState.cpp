/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_TimeLoopAcceptedState.cpp
 * @brief The accepted state under first-order generalized-alpha: time levels
 *        of essential data (stage and endpoint values, restart, startup
 *        rates), rollback of rejected candidates, and the initial rate.
 *
 * The system is M du/dt = 0 on the single tetrahedron with u_1 = g(t) and
 * u_2 = u_3 = 0.  The free row of u_0 is the consistent-mass row
 * 2 du_0/dt + dg/dt = 0, so u_0 = -g/2 when the data start consistent.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/LinearSolver.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/SystemConstraint.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"
#include "TimeStepping/StepController.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using svmp::FE::GlobalIndex;
using svmp::FE::Real;
namespace ts = svmp::FE::timestepping;
namespace ts_test = svmp::FE::timestepping::test;

/// u_1 = g(t), u_2 = u_3 = 0.  The setup value is g(0), as for a constraint
/// that does not know the time before the first update.
class PrescribedDofConstraint final : public svmp::FE::constraints::ISystemConstraint {
public:
    explicit PrescribedDofConstraint(std::function<double(double)> g) : g_(std::move(g)) {}

    void apply(const svmp::FE::systems::FESystem& /*system*/,
               svmp::FE::constraints::AffineConstraints& constraints) override
    {
        constraints.addDirichlet(1, g_(0.0));
        constraints.addDirichlet(2, 0.0);
        constraints.addDirichlet(3, 0.0);
    }

    bool updateValues(const svmp::FE::systems::FESystem& /*system*/,
                      svmp::FE::constraints::AffineConstraints& constraints,
                      double time,
                      double /*dt*/) override
    {
        constraints.updateInhomogeneity(1, g_(time));
        return true;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override { return true; }

    [[nodiscard]] svmp::FE::systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        return {};
    }

private:
    std::function<double(double)> g_;
};

/// Retries a rejected attempt with the same step.
class RetrySameStepController final : public ts::StepController {
public:
    [[nodiscard]] int maxRetries() const noexcept override { return 2; }
    [[nodiscard]] ts::StepDecision onAccepted(const ts::StepAttemptInfo& info) override
    {
        return ts::StepDecision{.accept = true, .retry = false, .next_dt = info.dt};
    }
    [[nodiscard]] ts::StepDecision onRejected(const ts::StepAttemptInfo& info,
                                              ts::StepRejectReason /*reason*/) override
    {
        return ts::StepDecision{.accept = false, .retry = true, .next_dt = info.dt};
    }
};

/// Accepted state of a restart: step index, u_n, u_{n-1} and du_n/dt.
struct RestartState {
    int step_index{0};
    std::vector<Real> u_n{};
    std::vector<Real> u_nm1{};
    std::vector<Real> u_dot_n{};
};

struct RunSetup {
    std::function<double(double)> g{};
    double t0{0.0};
    double dt{0.1};
    double t_end{0.1};
    std::optional<RestartState> restart{};
    std::function<void(ts::TimeLoopCallbacks&)> callbacks{};
    std::shared_ptr<ts::StepController> controller{};
};

struct RunResult {
    std::vector<Real> u{};
    std::vector<Real> u_dot{};
};

std::optional<RunResult> runMassRow(const RunSetup& setup)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    svmp::FE::forms::FormCompiler compiler;
    sys.addCellKernel("op", u_field, u_field,
                      std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
                          compiler.compileResidual((svmp::FE::forms::dt(u) * v).dx()),
                          svmp::FE::forms::ADMode::Forward));
    sys.addSystemConstraint(std::make_unique<PrescribedDofConstraint>(setup.g));
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    svmp::FE::systems::TransientSystem transient(
        sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());
    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available";
        return std::nullopt;
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    auto history = ts::TimeHistory::allocate(*factory, 4, 2);
    if (setup.restart.has_value()) {
        const auto& r = *setup.restart;
        ts_test::setVectorByDof(history.uPrevK(1), r.u_n);
        ts_test::setVectorByDof(history.uPrevK(2), r.u_nm1);
        history.resetCurrentToPrevious();
        history.ensureSecondOrderState(*factory);
        ts_test::setVectorByDof(history.uDot(), r.u_dot_n);
        history.setStepIndex(r.step_index);
    } else {
        for (int k = 1; k <= history.historyDepth(); ++k) {
            ts_test::setVectorByDof(history.uPrevK(k), {0.0, 0.0, 0.0, 0.0});
        }
        history.resetCurrentToPrevious();
    }
    history.setPrevDt(setup.dt);

    ts::TimeLoopOptions opts;
    opts.t0 = setup.t0;
    opts.t_end = setup.t_end;
    opts.dt = setup.dt;
    opts.max_steps = 1000;
    opts.scheme = ts::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-14;
    opts.newton.rel_tolerance = 0.0;
    opts.step_controller = setup.controller;

    ts::TimeLoopCallbacks callbacks;
    if (setup.callbacks) {
        setup.callbacks(callbacks);
    }
    ts::TimeLoop loop(opts);
    const auto report = loop.run(transient, *factory, *linear, history, callbacks);
    if (!report.success) {
        ADD_FAILURE() << "TimeLoop failed: " << report.message;
        return std::nullopt;
    }
    return RunResult{ts_test::getVectorByDof(history.u()), ts_test::getVectorByDof(history.uDot())};
}

// rho_inf = 0.5: alpha_m = 5/6, alpha_f = 2/3, gamma = 2/3.
constexpr double kAlphaM = 5.0 / 6.0;

TEST(TimeLoopAcceptedState, GeneralizedAlphaStageImposesTheInterpolantOfEndpointData)
{
    // b = t^2 on the mass row 2 a_dot + b_dot = 0.  Imposing g(t_n + alpha_f dt)
    // at the stage and g(t_{n+1}) at the endpoint gave errors in a(1) of
    // 0.0204861, 0.0103299 and 0.00518663 for dt = 0.1, 0.05, 0.025.  With the
    // interpolant (1 - alpha_f) g(t_n) + alpha_f g(t_{n+1}) at the stage the
    // linear relation 2a + b = 0 holds at every endpoint.
    for (const double dt : {0.1, 0.05, 0.025}) {
        SCOPED_TRACE(dt);
        const auto result = runMassRow({.g = [](double t) { return t * t; }, .dt = dt, .t_end = 1.0});
        ASSERT_TRUE(result.has_value());
        EXPECT_NEAR(result->u[0], -0.5, 1e-12);
        EXPECT_NEAR(result->u[1], 1.0, 1e-12);
    }
}

TEST(TimeLoopAcceptedState, GeneralizedAlphaStartupGivesConstrainedRatesTheirDataRate)
{
    // b = t from a zero start: the constrained rate starts at g_dot = 1.  A zero
    // start gave the first stage b_dot = alpha_m / gamma = 5/4, an endpoint rate
    // of 1.5, and a_dot(1) = -0.75.
    const auto result = runMassRow({.g = [](double t) { return t; }, .dt = 0.1, .t_end = 0.1});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->u_dot[1], 1.0, 1e-12);
    // The free row has a_dot(alpha_m) = -b_dot(alpha_m) / 2 = -1/2 from a_dot_0 = 0.
    EXPECT_NEAR(result->u_dot[0], -0.5 / kAlphaM, 1e-12);
}

TEST(TimeLoopAcceptedState, GeneralizedAlphaRestartUsesConstraintsAtTheRestoredTime)
{
    // Restart at t = 1 from the consistent state of b = t.  Stage entry used to
    // distribute the setup-time value g(0) = 0 into u_n, so the stage saw b jump
    // from 0; it also wrote g(t_n) = 1.0 into u_{n-1}, whose value is 0.9.
    std::optional<Real> b_nm1_at_gate;
    const auto result = runMassRow(
        {.g = [](double t) { return t; },
         .t0 = 1.0,
         .dt = 0.1,
         .t_end = 1.1,
         .restart = RestartState{.step_index = 10,
                                 .u_n = {-0.5, 1.0, 0.0, 0.0},
                                 .u_nm1 = {-0.45, 0.9, 0.0, 0.0},
                                 .u_dot_n = {-0.5, 1.0, 0.0, 0.0}},
         .callbacks = [&](ts::TimeLoopCallbacks& callbacks) {
             callbacks.on_step_candidate_ready =
                 [&](ts::TimeHistory& history, const ts::NewtonReport&) -> std::optional<ts::StepRejectReason> {
                 b_nm1_at_gate = ts_test::getVectorByDof(history.uPrev2())[1];
                 return std::nullopt;
             };
         }});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->u[0], -0.55, 1e-12);
    EXPECT_NEAR(result->u[1], 1.1, 1e-12);
    ASSERT_TRUE(b_nm1_at_gate.has_value());
    EXPECT_DOUBLE_EQ(*b_nm1_at_gate, 0.9);
}

TEST(TimeLoopAcceptedState, RejectedCandidateRestoresStoredSolutionsAndRatesBeforeObservers)
{
    // b = t^2, restarted at t = 1.  The first attempt is rejected after a value
    // has been written into u_n (as a stage solve projects trial constraint
    // lines into it).  The rejection observer must see the accepted u_n and
    // rate, and the retry must reproduce the run without a rejection.
    const RestartState restart{.step_index = 10,
                               .u_n = {-0.5, 1.0, 0.0, 0.0},
                               .u_nm1 = {-0.405, 0.81, 0.0, 0.0},
                               .u_dot_n = {-1.0, 2.0, 0.0, 0.0}};
    const auto g = [](double t) { return t * t; };
    const auto reference = runMassRow({.g = g, .t0 = 1.0, .dt = 0.1, .t_end = 1.1, .restart = restart});
    ASSERT_TRUE(reference.has_value());

    int attempts = 0;
    std::vector<Real> observed_u_n;
    std::vector<Real> observed_rate;
    const auto retried = runMassRow(
        {.g = g,
         .t0 = 1.0,
         .dt = 0.1,
         .t_end = 1.1,
         .restart = restart,
         .callbacks =
             [&](ts::TimeLoopCallbacks& callbacks) {
                 callbacks.on_before_step_accept = [&](ts::TimeHistory& history, const ts::NewtonReport&) {
                     if (attempts++ > 0) {
                         return true;
                     }
                     ts_test::setVectorByDof(history.uPrev(), {2.0, 1.0, 0.0, 0.0});
                     return false;
                 };
                 callbacks.on_step_rejected =
                     [&](const ts::TimeHistory& history, ts::StepRejectReason, const ts::NewtonReport&) {
                         observed_u_n = ts_test::getVectorByDof(const_cast<ts::TimeHistory&>(history).uPrev());
                         observed_rate = ts_test::getVectorByDof(const_cast<ts::TimeHistory&>(history).uDot());
                     };
             },
         .controller = std::make_shared<RetrySameStepController>()});
    ASSERT_TRUE(retried.has_value());
    EXPECT_EQ(attempts, 2);
    ASSERT_EQ(observed_u_n.size(), 4u);
    EXPECT_DOUBLE_EQ(observed_u_n[0], -0.5);
    ASSERT_EQ(observed_rate.size(), 4u);
    EXPECT_DOUBLE_EQ(observed_rate[0], -1.0);
    EXPECT_DOUBLE_EQ(observed_rate[1], 2.0);
    for (std::size_t i = 0; i < 4u; ++i) {
        EXPECT_DOUBLE_EQ(retried->u[i], reference->u[i]) << "dof=" << i;
        EXPECT_DOUBLE_EQ(retried->u_dot[i], reference->u_dot[i]) << "dof=" << i;
    }
}

/// Writes garbage into the solution and throws on its first solve, then
/// delegates.
class ThrowingFirstSolve final : public svmp::FE::backends::LinearSolver {
public:
    explicit ThrowingFirstSolve(svmp::FE::backends::LinearSolver& inner) : inner_(inner) {}
    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }
    void setOptions(const svmp::FE::backends::SolverOptions& options) override { inner_.setOptions(options); }
    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }
    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                         svmp::FE::backends::GenericVector& x,
                                                         const svmp::FE::backends::GenericVector& b) override
    {
        if (solves_++ == 0) {
            for (auto& value : x.localSpan()) {
                value = 7.0;
            }
            throw std::runtime_error("linear solve failed");
        }
        return inner_.solve(A, x, b);
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    int solves_{0};
};

/// One generalized-alpha step of du/dt = 1 from u = 0, with the initial rate
/// from the rate solve or from the solution history.
std::optional<std::vector<Real>> runUnitSourceStep(bool rate_from_pde, bool throw_first_solve)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    svmp::FE::forms::FormCompiler compiler;
    sys.addCellKernel("op", u_field, u_field,
                      std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
                          compiler.compileResidual((svmp::FE::forms::dt(u) * v).dx() -
                                                   (svmp::FE::forms::FormExpr::constant(1.0) * v).dx()),
                          svmp::FE::forms::ADMode::Forward));
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    svmp::FE::systems::TransientSystem transient(
        sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());
    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available";
        return std::nullopt;
    }
    auto direct = factory->createLinearSolver(ts_test::directSolve());
    ThrowingFirstSolve throwing(*direct);
    svmp::FE::backends::LinearSolver& linear = throw_first_solve
                                                   ? static_cast<svmp::FE::backends::LinearSolver&>(throwing)
                                                   : *direct;
    auto history = ts::TimeHistory::allocate(*factory, 4, 2);
    for (int k = 1; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), {0.0, 0.0, 0.0, 0.0});
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(0.1);

    ts::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.1;
    opts.dt = 0.1;
    opts.max_steps = 10;
    opts.scheme = ts::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.initialize_first_order_rate_from_pde = rate_from_pde;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-14;
    opts.newton.rel_tolerance = 0.0;
    ts::TimeLoop loop(opts);
    const auto report = loop.run(transient, *factory, linear, history, {});
    if (!report.success) {
        ADD_FAILURE() << "TimeLoop failed: " << report.message;
        return std::nullopt;
    }
    return ts_test::getVectorByDof(history.u());
}

TEST(TimeLoopAcceptedState, InitialRateSolveThatThrowsIsNotAccepted)
{
    // The rate solve gives u_dot = 1 and the increment 0.1.  When it throws,
    // the start falls back to the history rate 0 (increment 0.08); the thrown
    // solve's zero residual norms used to accept its partial rate 7
    // (increment 0.22).
    const auto exact_start = runUnitSourceStep(/*rate_from_pde=*/true, /*throw_first_solve=*/false);
    const auto history_start = runUnitSourceStep(/*rate_from_pde=*/false, /*throw_first_solve=*/false);
    const auto failed_solve = runUnitSourceStep(/*rate_from_pde=*/true, /*throw_first_solve=*/true);
    ASSERT_TRUE(exact_start.has_value() && history_start.has_value() && failed_solve.has_value());
    for (std::size_t i = 0; i < 4u; ++i) {
        EXPECT_NEAR((*exact_start)[i], 0.1, 1e-12);
        EXPECT_NEAR((*history_start)[i], 0.08, 1e-12);
        EXPECT_DOUBLE_EQ((*failed_solve)[i], (*history_start)[i]) << "dof=" << i;
    }
}

} // namespace
