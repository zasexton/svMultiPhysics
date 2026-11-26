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
 *        rates).
 *
 * The system is M du/dt = 0 on the single tetrahedron with u_1 = g(t) and
 * u_2 = u_3 = 0.  The free row of u_0 is the consistent-mass row
 * 2 du_0/dt + dg/dt = 0, so u_0 = -g/2 when the data start consistent.
 */

#include <gtest/gtest.h>

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
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
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
    // from 0.
    const auto result = runMassRow(
        {.g = [](double t) { return t; },
         .t0 = 1.0,
         .dt = 0.1,
         .t_end = 1.1,
         .restart = RestartState{.step_index = 10,
                                 .u_n = {-0.5, 1.0, 0.0, 0.0},
                                 .u_nm1 = {-0.45, 0.9, 0.0, 0.0},
                                 .u_dot_n = {-0.5, 1.0, 0.0, 0.0}}});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->u[0], -0.55, 1e-12);
    EXPECT_NEAR(result->u[1], 1.1, 1e-12);
}

} // namespace
