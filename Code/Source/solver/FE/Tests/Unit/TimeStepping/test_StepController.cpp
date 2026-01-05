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

#include "TimeStepping/StepController.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

void setVectorByDof(svmp::FE::backends::GenericVector& vec, const std::vector<Real>& values)
{
    ASSERT_EQ(static_cast<std::size_t>(vec.size()), values.size());
    auto view = vec.createAssemblyView();
    ASSERT_NE(view.get(), nullptr);
    view->beginAssemblyPhase();
    for (GlobalIndex i = 0; i < vec.size(); ++i) {
        view->addVectorEntry(i, values[static_cast<std::size_t>(i)], svmp::FE::assembly::AddMode::Insert);
    }
    view->finalizeAssembly();
}

std::unique_ptr<svmp::FE::backends::BackendFactory> createTestFactory()
{
#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    return svmp::FE::backends::BackendFactory::create(svmp::FE::backends::BackendKind::Eigen);
#else
    return nullptr;
#endif
}

svmp::FE::backends::SolverOptions directSolve()
{
    svmp::FE::backends::SolverOptions opts;
    opts.method = svmp::FE::backends::SolverMethod::Direct;
    opts.preconditioner = svmp::FE::backends::PreconditionerType::None;
    opts.rel_tol = 1e-14;
    opts.abs_tol = 1e-14;
    opts.max_iter = 1;
    return opts;
}

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
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
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
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
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
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto inner = factory->createLinearSolver(directSolve());
    ASSERT_NE(inner.get(), nullptr);
    FailOnceLinearSolver linear(*inner, /*failures=*/1);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
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
