/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"

#include "Spaces/L2Space.h"

#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ts_test = svmp::FE::timestepping::test;

namespace {

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
        inner_->computeInteriorFace(ctx_minus,
                                    ctx_plus,
                                    output_minus,
                                    output_plus,
                                    coupling_minus_plus,
                                    coupling_plus_minus);
    }

    [[nodiscard]] std::string name() const override
    {
        return "Counting(" + inner_->name() + ")";
    }

private:
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner_{};
    KernelCallCounts* counts_{nullptr};
};

class ScalingLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    ScalingLinearSolver(svmp::FE::backends::LinearSolver& inner, double scale)
        : inner_(inner)
        , scale_(scale)
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
        auto rep = inner_.solve(A, x, b);
        x.scale(static_cast<svmp::FE::Real>(scale_));
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    double scale_{1.0};
};

class AlwaysFailLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit AlwaysFailLinearSolver(svmp::FE::backends::LinearSolver& inner)
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
                                                          svmp::FE::backends::GenericVector&,
                                                          const svmp::FE::backends::GenericVector&) override
    {
        svmp::FE::backends::SolverReport rep;
        rep.converged = false;
        rep.iterations = 0;
        rep.message = "intentional test failure";
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
};

struct ScalarProblem {
    std::shared_ptr<svmp::FE::forms::test::SingleTetraMeshAccess> mesh{};
    std::shared_ptr<svmp::FE::spaces::L2Space> space{};
    std::unique_ptr<svmp::FE::systems::FESystem> sys{};
    svmp::FE::FieldId u_field{std::numeric_limits<svmp::FE::FieldId>::max()};
    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> integrator{};
    std::unique_ptr<svmp::FE::systems::TransientSystem> transient{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    std::unique_ptr<svmp::FE::backends::LinearSolver> linear{};
    svmp::FE::timestepping::TimeHistory history{};
};

template <typename BuildForm>
[[nodiscard]] ScalarProblem makeScalarProblem(BuildForm build_form,
                                              double dt,
                                              const std::vector<svmp::FE::Real>& u0,
                                              KernelCallCounts* counts = nullptr)
{
    ScalarProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*p.space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*p.space, "v");
    const auto form = build_form(u, v);

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto base_kernel =
        std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> kernel = base_kernel;
    if (counts != nullptr) {
        kernel = std::make_shared<CountingKernel>(kernel, counts);
    }
    p.sys->addCellKernel("op", p.u_field, p.u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(*p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error("ScalarProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("ScalarProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("ScalarProblem u0 size mismatch");
    }
    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] double scalarFromDofVector(svmp::FE::backends::GenericVector& vec)
{
    const auto vals = ts_test::getVectorByDof(vec);
    if (vals.size() != 1u) {
        throw std::runtime_error("Expected scalar DOF vector");
    }
    return static_cast<double>(vals[0]);
}

} // namespace

TEST(NewtonSolverLineSearch, BacktracksWhenFullStepIncreasesResidual)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_shrink = 0.5;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/3.0);
    (void)newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    // One initial residual assembly, then alpha=1 (reject) and alpha=0.5 (accept).
    EXPECT_EQ(counts.vector_only, 3);

    const double u_after = scalarFromDofVector(problem.history.u());
    EXPECT_NEAR(u_after, -0.5, 1e-13);
}

TEST(NewtonSolverLineSearch, ClampsAlphaToMinWhenShrinkWouldGoBelow)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 5;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 0.6;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Force an overshoot so alpha=1 fails, and even alpha_min still fails; the solver
    // must clamp to alpha_min and accept that last trial.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/4.0);
    (void)newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    // One initial residual assembly, then alpha=1 and alpha=alpha_min.
    EXPECT_EQ(counts.vector_only, 3);

    const double u_after = scalarFromDofVector(problem.history.u());
    EXPECT_NEAR(u_after, -1.4, 1e-13);
}

TEST(NewtonSolverLineSearch, StepToleranceUsesLastTriedAlphaWhenMaxIterationsReached)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.75;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 1e-12;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Flip the Newton direction so the Armijo condition can never be satisfied. When the
    // line search hits its max iterations, step_tolerance must use the last tried alpha.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    const auto rep = newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);
}

TEST(NewtonSolver, ReusesJacobianWhenRebuildPeriodGreaterThanOne)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 5;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = false;
    nopt.jacobian_rebuild_period = 3;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, nopt.max_iterations);

    // One residual assembly per Newton iteration, but Jacobian only on iterations 0 and 3.
    EXPECT_EQ(counts.vector_only, 5);
    EXPECT_EQ(counts.matrix_only, 2);
}

TEST(NewtonSolver, ScalesDtIncrementsByDtOrExplicitFactor)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;
    const std::vector<svmp::FE::Real> u0 = {1.0};

    auto problem = makeScalarProblem(
        [&](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            return (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();
        },
        dt,
        u0);

    const double u_exact = 1.0 / (1.0 + lambda * dt);
    const double du = 1.0 - u_exact;

    auto run_once = [&](bool scale_dt_increments, double dt_increment_scale) -> double {
        ts_test::setVectorByDof(problem.history.uPrev(), u0);
        ts_test::setVectorByDof(problem.history.uPrev2(), u0);
        problem.history.resetCurrentToPrevious();

        svmp::FE::timestepping::NewtonOptions nopt;
        nopt.residual_op = "op";
        nopt.jacobian_op = "op";
        nopt.max_iterations = 1;
        nopt.abs_tolerance = 0.0;
        nopt.rel_tolerance = 0.0;
        nopt.step_tolerance = 0.0;
        nopt.use_line_search = false;
        nopt.scale_dt_increments = scale_dt_increments;
        nopt.dt_increment_scale = dt_increment_scale;

        svmp::FE::timestepping::NewtonSolver newton(nopt);
        svmp::FE::timestepping::NewtonWorkspace ws;
        newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
        problem.history.repack(*problem.factory);

        (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
        return scalarFromDofVector(problem.history.u());
    };

    const double u_unscaled = run_once(/*scale_dt_increments=*/false, /*dt_increment_scale=*/0.0);
    EXPECT_NEAR(u_unscaled, u_exact, 1e-13);

    const double u_scaled_by_dt = run_once(/*scale_dt_increments=*/true, /*dt_increment_scale=*/0.0);
    EXPECT_NEAR(u_scaled_by_dt, 1.0 - dt * du, 1e-13);

    const double u_scaled_explicit = run_once(/*scale_dt_increments=*/true, /*dt_increment_scale=*/0.5);
    EXPECT_NEAR(u_scaled_explicit, 1.0 - 0.5 * du, 1e-13);
}

TEST(NewtonSolver, ExhibitsQuadraticConvergenceNearSolution)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const double u_star = std::sqrt(2.0);

    auto u_val = [&]() { return scalarFromDofVector(problem.history.u()); };
    auto err = [&]() { return std::abs(u_val() - u_star); };

    const double e0 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e1 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e2 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e3 = err();

    EXPECT_LT(e1, e0);
    EXPECT_LT(e2, e1);
    EXPECT_LT(e3, e2);

    constexpr double C = 0.6;
    EXPECT_LE(e2, C * e1 * e1);
    EXPECT_LE(e3, C * e2 * e2);
}

TEST(NewtonSolver, ModifiedNewtonConvergesMoreSlowlyThanFullNewton)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto make_history = [&](const svmp::FE::backends::BackendFactory& factory,
                            svmp::FE::GlobalIndex n_dofs,
                            const std::vector<svmp::FE::Real>& u0) {
        auto history = svmp::FE::timestepping::TimeHistory::allocate(factory, n_dofs);
        history.setDt(0.1);
        history.setPrevDt(0.1);
        ts_test::setVectorByDof(history.uPrev(), u0);
        ts_test::setVectorByDof(history.uPrev2(), u0);
        history.resetCurrentToPrevious();
        return history;
    };

    auto base_problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    const auto n_dofs = base_problem.sys->dofHandler().getNumDofs();
    const std::vector<svmp::FE::Real> u0 = {1.5};

    auto history_full = make_history(*base_problem.factory, n_dofs, u0);
    auto history_mod = make_history(*base_problem.factory, n_dofs, u0);

    svmp::FE::timestepping::NewtonOptions full;
    full.residual_op = "op";
    full.jacobian_op = "op";
    full.max_iterations = 3;
    full.abs_tolerance = 0.0;
    full.rel_tolerance = 0.0;
    full.step_tolerance = 0.0;
    full.use_line_search = false;
    full.jacobian_rebuild_period = 1;

    svmp::FE::timestepping::NewtonOptions mod = full;
    mod.jacobian_rebuild_period = 100;

    svmp::FE::timestepping::NewtonSolver newton_full(full);
    svmp::FE::timestepping::NewtonSolver newton_mod(mod);

    svmp::FE::timestepping::NewtonWorkspace ws_full;
    svmp::FE::timestepping::NewtonWorkspace ws_mod;
    newton_full.allocateWorkspace(*base_problem.sys, *base_problem.factory, ws_full);
    newton_mod.allocateWorkspace(*base_problem.sys, *base_problem.factory, ws_mod);
    history_full.repack(*base_problem.factory);
    history_mod.repack(*base_problem.factory);

    (void)newton_full.solveStep(*base_problem.transient, *base_problem.linear, /*solve_time=*/history_full.dt(), history_full, ws_full);
    (void)newton_mod.solveStep(*base_problem.transient, *base_problem.linear, /*solve_time=*/history_mod.dt(), history_mod, ws_mod);

    const double u_star = std::sqrt(2.0);
    const double err_full = std::abs(scalarFromDofVector(history_full.u()) - u_star);
    const double err_mod = std::abs(scalarFromDofVector(history_mod.u()) - u_star);

    EXPECT_LT(err_full, 1e-10);
    EXPECT_GT(err_mod, 1e-7);
}

TEST(NewtonSolver, ReportContainsResidualNormsWhenNotConverged)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-20;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 2;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);
    EXPECT_TRUE(std::isfinite(rep.residual_norm0));
    EXPECT_TRUE(std::isfinite(rep.residual_norm));
    EXPECT_GT(rep.residual_norm0, 0.0);
    EXPECT_GT(rep.residual_norm0, rep.residual_norm);
    EXPECT_TRUE(rep.linear.converged);
}

TEST(NewtonSolver, ThrowsWhenLinearSolveFails)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    AlwaysFailLinearSolver failing(*problem.linear);
    EXPECT_THROW((void)newton.solveStep(*problem.transient,
                                        failing,
                                        /*solve_time=*/problem.history.dt(),
                                        problem.history,
                                        ws),
                 svmp::FE::FEException);
}
