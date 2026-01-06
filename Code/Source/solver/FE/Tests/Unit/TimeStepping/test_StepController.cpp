/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/LinearSolver.h"

#include "Core/FEException.h"
#include "Core/Types.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace ts_test = svmp::FE::timestepping::test;

namespace {

class RejectImmediatelyController final : public svmp::FE::timestepping::StepController {
public:
    [[nodiscard]] int maxRetries() const noexcept override { return 0; }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onAccepted(const svmp::FE::timestepping::StepAttemptInfo&) override
    {
        return {};
    }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onRejected(const svmp::FE::timestepping::StepAttemptInfo&,
               svmp::FE::timestepping::StepRejectReason) override
    {
        svmp::FE::timestepping::StepDecision d;
        d.accept = false;
        d.retry = false;
        d.next_dt = 0.0;
        d.message = "stop";
        return d;
    }
};

class HalveDtOnAcceptController final : public svmp::FE::timestepping::StepController {
public:
    [[nodiscard]] int maxRetries() const noexcept override { return 0; }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onAccepted(const svmp::FE::timestepping::StepAttemptInfo& info) override
    {
        svmp::FE::timestepping::StepDecision d;
        d.accept = true;
        d.retry = false;
        d.next_dt = 0.5 * info.dt;
        return d;
    }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onRejected(const svmp::FE::timestepping::StepAttemptInfo&,
               svmp::FE::timestepping::StepRejectReason) override
    {
        svmp::FE::timestepping::StepDecision d;
        d.accept = false;
        d.retry = false;
        return d;
    }
};

class HalveDtOnRejectController final : public svmp::FE::timestepping::StepController {
public:
    explicit HalveDtOnRejectController(int max_retries)
        : max_retries_(max_retries)
    {
    }

    [[nodiscard]] int maxRetries() const noexcept override { return max_retries_; }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onAccepted(const svmp::FE::timestepping::StepAttemptInfo&) override
    {
        return {};
    }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onRejected(const svmp::FE::timestepping::StepAttemptInfo& info,
               svmp::FE::timestepping::StepRejectReason) override
    {
        svmp::FE::timestepping::StepDecision d;
        d.accept = false;
        d.retry = true;
        d.next_dt = 0.5 * info.dt;
        return d;
    }

private:
    int max_retries_{0};
};

class FailOnceLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit FailOnceLinearSolver(svmp::FE::backends::LinearSolver& inner, int failures)
        : inner_(inner)
        , failures_remaining_(failures)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        if (failures_remaining_ > 0) {
            failures_remaining_ -= 1;
            svmp::FE::backends::SolverReport rep;
            rep.converged = false;
            rep.iterations = 0;
            rep.message = "forced failure for retry-path test";
            return rep;
        }
        return inner_.solve(A, x, b);
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    int failures_remaining_{0};
};

class RejectOnAcceptController final : public svmp::FE::timestepping::StepController {
public:
    [[nodiscard]] int maxRetries() const noexcept override { return 0; }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onAccepted(const svmp::FE::timestepping::StepAttemptInfo&) override
    {
        svmp::FE::timestepping::StepDecision d;
        d.accept = false;
        d.retry = false;
        d.next_dt = 0.0;
        d.message = "reject for callback test";
        return d;
    }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onRejected(const svmp::FE::timestepping::StepAttemptInfo&,
               svmp::FE::timestepping::StepRejectReason) override
    {
        return {};
    }
};

} // namespace

TEST(TimeLoopAdaptiveStep, ReturnsFailureInsteadOfThrowWhenControllerStops)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.2;
    opts.dt = 0.2;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 1; // Force failure even on linear problems.
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    opts.step_controller = std::make_shared<RejectImmediatelyController>();

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    int rejected = 0;
    callbacks.on_step_rejected = [&rejected](const svmp::FE::timestepping::TimeHistory&,
                                             svmp::FE::timestepping::StepRejectReason,
                                             const svmp::FE::timestepping::NewtonReport&) {
        rejected += 1;
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::TimeLoopReport rep;
    ASSERT_NO_THROW(rep = loop.run(transient, *factory, *linear, history, callbacks));

    EXPECT_FALSE(rep.success);
    EXPECT_EQ(rejected, 1);
    EXPECT_NEAR(history.time(), 0.0, 1e-15);
}

TEST(TimeLoopAdaptiveStep, CallsDtUpdatedCallbackOnAcceptedStep)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.3;
    opts.dt = 0.2;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    opts.step_controller = std::make_shared<HalveDtOnAcceptController>();

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    std::vector<std::pair<double, double>> dt_updates;
    callbacks.on_dt_updated = [&dt_updates](double oldv, double newv, int, int) {
        dt_updates.emplace_back(oldv, newv);
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, callbacks);
    EXPECT_TRUE(rep.success);
    ASSERT_EQ(dt_updates.size(), 2U);
    EXPECT_NEAR(dt_updates[0].first, 0.2, 1e-14);
    EXPECT_NEAR(dt_updates[0].second, 0.1, 1e-14);
    EXPECT_NEAR(dt_updates[1].first, 0.1, 1e-14);
    EXPECT_NEAR(dt_updates[1].second, 0.05, 1e-14);
    EXPECT_NEAR(rep.final_time, 0.3, 1e-12);
}

TEST(TimeLoopAdaptiveStep, RetriesWhenLinearSolveFailsThenRecovers)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    // Linear scalar ODE: u' + u = 0.
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto inner = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(inner.get(), nullptr);
    FailOnceLinearSolver linear(*inner, /*failures=*/1);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.5;
    opts.dt = 0.5;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    opts.step_controller = std::make_shared<HalveDtOnRejectController>(6);

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    int rejected = 0;
    callbacks.on_step_rejected = [&rejected](const svmp::FE::timestepping::TimeHistory&,
                                             svmp::FE::timestepping::StepRejectReason,
                                             const svmp::FE::timestepping::NewtonReport&) {
        rejected += 1;
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, linear, history, callbacks);
    EXPECT_TRUE(rep.success);
    EXPECT_GE(rejected, 1);
    EXPECT_NEAR(rep.final_time, 0.5, 1e-12);
}

TEST(TimeLoopOptionsValidation, ConstructorValidatesTimeLoopOptions)
{
    using svmp::FE::timestepping::SchemeKind;
    using svmp::FE::timestepping::TimeLoop;
    using svmp::FE::timestepping::TimeLoopOptions;

    auto base = [] {
        TimeLoopOptions o;
        o.t0 = 0.0;
        o.t_end = 1.0;
        o.dt = 0.1;
        o.max_steps = 5;
        return o;
    };

    {
        auto o = base();
        o.dt = 0.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.dt = -1.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.t0 = 1.0;
        o.t_end = 0.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.max_steps = 0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::ThetaMethod;
        o.theta = -0.1;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::ThetaMethod;
        o.theta = 1.1;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::TRBDF2;
        o.trbdf2_gamma = 0.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::TRBDF2;
        o.trbdf2_gamma = 1.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::Newmark;
        o.newmark_beta = 0.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::Newmark;
        o.newmark_gamma = 0.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::DG;
        o.dg_degree = -1;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::DG;
        o.dg_degree = 11;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::CG;
        o.cg_degree = 0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::CG;
        o.cg_degree = 11;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::DG1;
        o.collocation_max_outer_iterations = 0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::CG2;
        o.collocation_outer_tolerance = -1.0;
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
    {
        auto o = base();
        o.scheme = SchemeKind::VSVO_BDF;
        o.step_controller.reset();
        EXPECT_THROW((void)TimeLoop{o}, svmp::FE::InvalidArgumentException);
    }
}

TEST(TimeLoopCallbacks, CallsOnStepStartNonlinearDoneAndStepAccepted)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.2;
    opts.dt = 0.2;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    int step_start_calls = 0;
    int nonlinear_done_calls = 0;
    int step_accepted_calls = 0;
    bool last_nonlinear_converged = false;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_step_start = [&step_start_calls](const svmp::FE::timestepping::TimeHistory&) { step_start_calls += 1; };
    callbacks.on_nonlinear_done = [&nonlinear_done_calls, &last_nonlinear_converged](const svmp::FE::timestepping::TimeHistory&,
                                                                                     const svmp::FE::timestepping::NewtonReport& nr) {
        nonlinear_done_calls += 1;
        last_nonlinear_converged = nr.converged;
    };
    callbacks.on_step_accepted = [&step_accepted_calls](const svmp::FE::timestepping::TimeHistory&) { step_accepted_calls += 1; };

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, callbacks);
    EXPECT_TRUE(rep.success);
    EXPECT_EQ(step_start_calls, 1);
    EXPECT_EQ(nonlinear_done_calls, 1);
    EXPECT_TRUE(last_nonlinear_converged);
    EXPECT_EQ(step_accepted_calls, 1);
}

TEST(TimeLoopCallbacks, CallsStepRejectedWithErrorTooLargeReasonWhenControllerRejectsAcceptedStep)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.2;
    opts.dt = 0.2;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    opts.step_controller = std::make_shared<RejectOnAcceptController>();

    int rejected_calls = 0;
    std::optional<svmp::FE::timestepping::StepRejectReason> last_reason;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_step_rejected = [&rejected_calls, &last_reason](const svmp::FE::timestepping::TimeHistory&,
                                                                 svmp::FE::timestepping::StepRejectReason reason,
                                                                 const svmp::FE::timestepping::NewtonReport&) {
        rejected_calls += 1;
        last_reason = reason;
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, callbacks);
    EXPECT_FALSE(rep.success);
    EXPECT_EQ(rejected_calls, 1);
    ASSERT_TRUE(last_reason.has_value());
    EXPECT_EQ(*last_reason, svmp::FE::timestepping::StepRejectReason::ErrorTooLarge);
    EXPECT_NEAR(history.time(), 0.0, 1e-15);
}

TEST(TimeLoopEdgeCases, ZeroDurationRunReturnsImmediateSuccess)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.0;
    opts.dt = 0.1;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history);
    EXPECT_TRUE(rep.success);
    EXPECT_EQ(rep.steps_taken, 0);
    EXPECT_NEAR(rep.final_time, 0.0, 1e-15);
    EXPECT_NEAR(history.time(), 0.0, 1e-15);
}

TEST(TimeLoopEdgeCases, AdjustLastStepClampsDtToRemainingInterval)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.05;
    opts.dt = 0.1;
    opts.adjust_last_step = true;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history);
    EXPECT_TRUE(rep.success);
    EXPECT_EQ(rep.steps_taken, 1);
    EXPECT_NEAR(rep.final_time, 0.05, 1e-12);
    EXPECT_NEAR(history.time(), 0.05, 1e-12);
    EXPECT_NEAR(history.dtPrev(), 0.05, 1e-12);
}
