/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/TimeIntegrationContext.h"
#include "Core/FEException.h"

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"

#include "Spaces/H1Space.h"

#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "TimeStepping/MultiStageScheme.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ts_test = svmp::FE::timestepping::test;

namespace {

class FakeIntegrator final : public svmp::FE::systems::TimeIntegrator {
public:
    [[nodiscard]] std::string name() const override { return "Fake"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

    [[nodiscard]] svmp::FE::assembly::TimeIntegrationContext
    buildContext(int /*max_time_derivative_order*/, const svmp::FE::systems::SystemStateView& /*state*/) const override
    {
        svmp::FE::assembly::TimeIntegrationContext ctx;
        ctx.integrator_name = name();
        ctx.time_derivative_term_weight = static_cast<svmp::FE::Real>(0.125);
        ctx.non_time_derivative_term_weight = static_cast<svmp::FE::Real>(0.25);
        svmp::FE::assembly::TimeDerivativeStencil s;
        s.order = 1;
        s.a = {1.0, -1.0};
        ctx.dt1 = s;
        return ctx;
    }
};

class ZeroUpdateLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit ZeroUpdateLinearSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
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

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix&,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector&) override
    {
        x.zero();
        svmp::FE::backends::SolverReport rep;
        rep.converged = true;
        rep.iterations = 0;
        rep.message = "forced zero update";
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
};

struct KernelCallCounts {
    int total{0};
    int matrix_only{0};
    int vector_only{0};
    int matrix_and_vector{0};
};

class CountingKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    CountingKernel(std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner, KernelCallCounts* counts)
        : inner_(std::move(inner))
        , counts_(counts)
    {
        if (!inner_) {
            throw std::runtime_error("CountingKernel: inner is null");
        }
        if (!counts_) {
            throw std::runtime_error("CountingKernel: counts is null");
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
        counts_->total += 1;
        if (output.has_matrix && output.has_vector) {
            counts_->matrix_and_vector += 1;
        } else if (output.has_matrix) {
            counts_->matrix_only += 1;
        } else if (output.has_vector) {
            counts_->vector_only += 1;
        }
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
        return "Counting(" + inner_->name() + ")";
    }

private:
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner_{};
    KernelCallCounts* counts_{nullptr};
};

} // namespace

TEST(WeightedIntegrator, OverridesTermWeightsButPreservesStencils)
{
    auto base = std::make_shared<const FakeIntegrator>();
    svmp::FE::timestepping::WeightedIntegrator weighted(base,
                                                        static_cast<svmp::FE::Real>(3.0),
                                                        static_cast<svmp::FE::Real>(4.0));

    svmp::FE::systems::SystemStateView state;
    const auto ctx = weighted.buildContext(/*max_time_derivative_order=*/1, state);

    EXPECT_EQ(ctx.integrator_name, "Fake");
    ASSERT_TRUE(ctx.dt1.has_value());
    EXPECT_EQ(ctx.dt1->order, 1);
    ASSERT_EQ(ctx.dt1->a.size(), 2u);
    EXPECT_NEAR(ctx.dt1->a[0], 1.0, 1e-15);
    EXPECT_NEAR(ctx.dt1->a[1], -1.0, 1e-15);
    EXPECT_NEAR(ctx.time_derivative_term_weight, 3.0, 1e-15);
    EXPECT_NEAR(ctx.non_time_derivative_term_weight, 4.0, 1e-15);
}

TEST(NewtonSolver, ValidatesOptions)
{
    using svmp::FE::timestepping::NewtonOptions;
    using svmp::FE::timestepping::NewtonSolver;

    {
        NewtonOptions o;
        o.max_iterations = 0;
        EXPECT_THROW((void)NewtonSolver{o}, svmp::FE::InvalidArgumentException);
    }
    {
        NewtonOptions o;
        o.abs_tolerance = -1.0;
        EXPECT_THROW((void)NewtonSolver{o}, svmp::FE::InvalidArgumentException);
    }
    {
        NewtonOptions o;
        o.rel_tolerance = -1.0;
        EXPECT_THROW((void)NewtonSolver{o}, svmp::FE::InvalidArgumentException);
    }
    {
        NewtonOptions o;
        o.step_tolerance = -1.0;
        EXPECT_THROW((void)NewtonSolver{o}, svmp::FE::InvalidArgumentException);
    }
}

TEST(NewtonSolver, ThrowsWhenWorkspaceNotAllocated)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();

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
    history.setDt(dt);
    history.setPrevDt(dt);
    const std::vector<svmp::FE::Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::NewtonSolver newton;
    svmp::FE::timestepping::NewtonWorkspace ws;
    EXPECT_FALSE(ws.isAllocated());

    EXPECT_THROW((void)newton.solveStep(transient, *linear, /*solve_time=*/dt, history, ws),
                 svmp::FE::InvalidArgumentException);
}

TEST(NewtonSolver, ThrowsOnInvalidDtOrSolveTime)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();

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
    history.setDt(dt);
    history.setPrevDt(dt);
    const std::vector<svmp::FE::Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 2;
    nopt.abs_tolerance = 1e-12;
    nopt.rel_tolerance = 0.0;
    svmp::FE::timestepping::NewtonSolver newton(nopt);

    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);

    // Match TimeLoop: ensure history vectors use backend layout created after workspace.
    history.repack(*factory);

    history.setDt(0.0);
    EXPECT_THROW((void)newton.solveStep(transient, *linear, /*solve_time=*/dt, history, ws),
                 svmp::FE::InvalidArgumentException);

    history.setDt(dt);
    EXPECT_THROW((void)newton.solveStep(transient, *linear, /*solve_time=*/std::numeric_limits<double>::infinity(), history, ws),
                 svmp::FE::InvalidArgumentException);
}

TEST(NewtonSolver, RelativeTolerancePreventsEarlyExitAtInitialResidual)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();

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
    history.setDt(dt);
    history.setPrevDt(dt);
    const std::vector<svmp::FE::Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 5;
    nopt.abs_tolerance = 1e100; // abs_ok at it=0
    nopt.rel_tolerance = 0.5;   // rel_ok false at it=0, forces at least one update
    svmp::FE::timestepping::NewtonSolver newton(nopt);

    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);
    history.repack(*factory);

    const auto rep = newton.solveStep(transient, *linear, /*solve_time=*/dt, history, ws);
    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);
}

TEST(NewtonSolver, StepToleranceCanConvergeOnZeroUpdate)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();

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
    ZeroUpdateLinearSolver linear(*inner);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    history.setDt(dt);
    history.setPrevDt(dt);
    const std::vector<svmp::FE::Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 5;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 1e-12;
    svmp::FE::timestepping::NewtonSolver newton(nopt);

    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);
    history.repack(*factory);

    const auto before = ts_test::getVectorByDof(history.uPrev());
    const auto rep = newton.solveStep(transient, linear, /*solve_time=*/dt, history, ws);
    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);

    // Forced zero update keeps the current iterate unchanged.
    const auto after = ts_test::getVectorByDof(history.u());
    EXPECT_EQ(after, before);
}

TEST(NewtonSolver, CanAssembleMatrixAndVectorSeparatelyWhenRequested)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;

    KernelCallCounts counts;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto inner_kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    auto kernel = std::make_shared<CountingKernel>(inner_kernel, &counts);
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
    history.setDt(dt);
    history.setPrevDt(dt);
    const std::vector<svmp::FE::Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    // First, allow combined assembly (single pass).
    {
        counts = KernelCallCounts{};
        svmp::FE::timestepping::NewtonOptions nopt;
        nopt.residual_op = "op";
        nopt.jacobian_op = "op";
        nopt.max_iterations = 1;
        nopt.abs_tolerance = 1e100;
        nopt.rel_tolerance = 0.0;
        nopt.assemble_both_when_possible = true;
        svmp::FE::timestepping::NewtonSolver newton(nopt);

        svmp::FE::timestepping::NewtonWorkspace ws;
        newton.allocateWorkspace(sys, *factory, ws);
        history.repack(*factory);

        const auto rep = newton.solveStep(transient, *linear, /*solve_time=*/dt, history, ws);
        EXPECT_TRUE(rep.converged);
        EXPECT_EQ(counts.total, 1);
        EXPECT_EQ(counts.matrix_and_vector, 1);
    }

    // Then, force separate assembly passes.
    {
        counts = KernelCallCounts{};
        svmp::FE::timestepping::NewtonOptions nopt;
        nopt.residual_op = "op";
        nopt.jacobian_op = "op";
        nopt.max_iterations = 1;
        nopt.abs_tolerance = 1e100;
        nopt.rel_tolerance = 0.0;
        nopt.assemble_both_when_possible = false;
        svmp::FE::timestepping::NewtonSolver newton(nopt);

        svmp::FE::timestepping::NewtonWorkspace ws;
        newton.allocateWorkspace(sys, *factory, ws);
        history.repack(*factory);

        const auto rep = newton.solveStep(transient, *linear, /*solve_time=*/dt, history, ws);
        EXPECT_TRUE(rep.converged);
        EXPECT_EQ(counts.total, 2);
    }
}

TEST(MultiStageSolver, ResidualAdditionShiftsLinearSolutionAsExpected)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "MultiStageScheme tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    history.setDt(dt);
    history.setPrevDt(dt);

    const std::vector<svmp::FE::Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 8;
    nopt.abs_tolerance = 1e-12;
    nopt.rel_tolerance = 0.0;
    svmp::FE::timestepping::NewtonSolver newton(nopt);

    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);
    ASSERT_TRUE(ws.isAllocated());

    // Match TimeLoop: ensure history vectors use backend layout created after workspace.
    history.repack(*factory);

    svmp::FE::timestepping::MultiStageSolver stages(newton);

    auto bdf1 = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();

    svmp::FE::timestepping::ImplicitStageSpec stage;
    stage.integrator = bdf1;
    stage.weights.time_derivative = static_cast<svmp::FE::Real>(1.0);
    stage.weights.non_time_derivative = static_cast<svmp::FE::Real>(1.0);
    stage.solve_time = dt;

    // Build a residual addition of the form g = M*c by assembling the same operator
    // at a state with dt(u)=0 and u = c/lambda => residual = lambda*M*u = M*c.
    const std::vector<svmp::FE::Real> c = {0.1, -0.2, 0.3, -0.4};
    std::vector<svmp::FE::Real> u_add(c.size(), 0.0);
    for (std::size_t i = 0; i < c.size(); ++i) {
        u_add[i] = static_cast<svmp::FE::Real>(static_cast<double>(c[i]) / lambda);
    }

    svmp::FE::timestepping::ResidualAdditionSpec add;
    add.integrator = bdf1;
    add.weights.time_derivative = static_cast<svmp::FE::Real>(1.0);
    add.weights.non_time_derivative = static_cast<svmp::FE::Real>(1.0);

    svmp::FE::systems::SystemStateView add_state;
    add_state.time = 0.0;
    add_state.dt = dt;
    add_state.dt_prev = dt;
    add_state.u = u_add;
    add_state.u_prev = u_add;
    add_state.u_prev2 = u_add;
    add.state = add_state;
    stage.residual_addition = add;

    auto scratch = factory->createVector(sys.dofHandler().getNumDofs());
    ASSERT_NE(scratch.get(), nullptr);

    const auto rep = stages.solveImplicitStage(sys, *linear, history, ws, stage, scratch.get());
    EXPECT_TRUE(rep.converged);

    const auto u_sol = ts_test::getVectorByDof(history.u());
    ASSERT_EQ(u_sol.size(), u0.size());

    const double denom = 1.0 + lambda * dt;
    for (std::size_t i = 0; i < u0.size(); ++i) {
        const double expected = static_cast<double>(u0[i]) / denom - (dt / denom) * static_cast<double>(c[i]);
        EXPECT_NEAR(static_cast<double>(u_sol[i]), expected, 5e-12);
    }
}
