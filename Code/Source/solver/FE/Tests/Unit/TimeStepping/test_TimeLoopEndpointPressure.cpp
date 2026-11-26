/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_TimeLoopEndpointPressure.cpp
 * @brief Endpoint values of algebraic fields under first-order
 *        generalized-alpha (free-surface audit A28 and R2).
 *
 * A declared algebraic interpolant p (fluid pressure) is recovered from its
 * converged stage value as p_{n+1} = (p_stage - (1 - alpha_f) p_n) / alpha_f
 * (Liu, Lan, Tikenogullari and Marsden, scheme 2).  The endpoint error obeys
 * e_{n+1} = -rho_inf e_n + delta_n / alpha_f: a step in the stage value
 * overshoots and decays alternately, a one-step stage pulse keeps its time
 * integral, and a constraint imposed at the stage leaves the endpoint with
 * B^{n+1} u_{n+1} = -rho_inf B^{n+1} u_n after a change of the constraint.
 */

#include <gtest/gtest.h>

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace {

using svmp::FE::Real;
namespace forms = svmp::FE::forms;
namespace ts = svmp::FE::timestepping;
namespace ts_test = svmp::FE::timestepping::test;

/// Endpoint and stage values of the algebraic field p of
/// du/dt = 0, p = f(t) on the single tetrahedron (P0 pressure).
struct PressureRun {
    std::vector<double> endpoint_times{};
    std::vector<double> endpoint_p{};
    std::vector<double> stage_times{};
    std::vector<double> stage_p{};
};

std::optional<PressureRun> runStagePressure(double rho_inf,
                                            double dt,
                                            double t_end,
                                            double p0,
                                            const std::function<double(double)>& f)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto u_space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);
    auto p_space = std::make_shared<svmp::FE::spaces::L2Space>(svmp::FE::ElementType::Tetra4, 0);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_id = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 1});
    const auto p_id = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("op");
    sys.attachment<ts::GeneralizedAlphaEndpointFields>().algebraic_fields.insert(p_id);

    const auto u = forms::FormExpr::trialFunction(*u_space, "u");
    const auto v = forms::FormExpr::testFunction(*u_space, "v");
    const auto p = forms::FormExpr::trialFunction(*p_space, "p");
    const auto q = forms::FormExpr::testFunction(*p_space, "q");
    const auto source = forms::FormExpr::coefficient(
        "stage_pressure", forms::TimeScalarCoefficient([f](Real, Real, Real, Real t) {
            return static_cast<Real>(f(static_cast<double>(t)));
        }));
    forms::FormCompiler compiler;
    sys.addCellKernel("op", u_id, u_id,
                      std::make_shared<forms::NonlinearFormKernel>(
                          compiler.compileResidual((forms::dt(u) * v).dx()), forms::ADMode::Forward));
    sys.addCellKernel("op", p_id, p_id,
                      std::make_shared<forms::NonlinearFormKernel>(
                          compiler.compileResidual(((p - source) * q).dx()), forms::ADMode::Forward));
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
    const auto n_dofs = sys.dofHandler().getNumDofs();
    const auto p_dof = static_cast<std::size_t>(sys.fieldDofOffset(p_id));
    auto history = ts::TimeHistory::allocate(*factory, n_dofs, 2);
    std::vector<Real> initial(static_cast<std::size_t>(n_dofs), 1.0);
    initial[p_dof] = static_cast<Real>(p0);
    for (int k = 1; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), initial);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    ts::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 10000;
    opts.scheme = ts::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = rho_inf;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-14;
    opts.newton.rel_tolerance = 0.0;

    PressureRun run;
    ts::TimeLoopCallbacks callbacks;
    // The converged stage and its time are available to observers before
    // the endpoint recovery.
    callbacks.on_candidate_stage = [&run, p_dof](const ts::CandidateStageObservation& stage) {
        ASSERT_NE(stage.state_vector, nullptr);
        run.stage_times.push_back(stage.state_time);
        run.stage_p.push_back(static_cast<double>(stage.state_vector->localSpan()[p_dof]));
    };
    callbacks.on_step_accepted = [&run, p_dof](ts::TimeHistory& accepted) {
        run.endpoint_times.push_back(accepted.time());
        run.endpoint_p.push_back(static_cast<double>(accepted.u().localSpan()[p_dof]));
    };
    ts::TimeLoop loop(opts);
    const auto report = loop.run(transient, *factory, *linear, history, callbacks);
    if (!report.success) {
        ADD_FAILURE() << "TimeLoop failed: " << report.message;
        return std::nullopt;
    }
    return run;
}

TEST(TimeLoopEndpointPressure, StagePressureStepOvershootsAndDecaysWithPoleMinusRhoInf)
{
    // The stage value steps from P to P + Delta between the third and the
    // fourth stage.  At rho_inf = 0.5 (alpha_f = 2/3) the endpoint values are
    // P + 1.5 Delta, P + 0.75 Delta, P + 1.125 Delta, P + 0.9375 Delta; at
    // rho_inf = 1 the overshoot alternates without decay.
    constexpr double P = 2.0;
    constexpr double Delta = 1.0;
    const auto step = [](double t) { return t < 0.33 ? P : P + Delta; };
    for (const auto& [rho_inf, expected] :
         std::vector<std::pair<double, std::vector<double>>>{{0.5, {1.5, 0.75, 1.125, 0.9375}},
                                                              {1.0, {2.0, 0.0, 2.0, 0.0}}}) {
        SCOPED_TRACE(rho_inf);
        const auto run = runStagePressure(rho_inf, 0.1, 0.7, P, step);
        ASSERT_TRUE(run.has_value());
        ASSERT_EQ(run->endpoint_p.size(), 7u);
        for (std::size_t n = 0; n < 3u; ++n) {
            EXPECT_NEAR(run->endpoint_p[n], P, 1e-12) << "step " << n;
        }
        for (std::size_t k = 0; k < expected.size(); ++k) {
            EXPECT_NEAR(run->endpoint_p[3u + k], P + expected[k] * Delta, 1e-12) << "step " << 3u + k;
        }
        // The stage value carries the step itself, stamped with its own time.
        const double alpha_f = 1.0 / (1.0 + rho_inf);
        ASSERT_EQ(run->stage_p.size(), 7u);
        EXPECT_NEAR(run->stage_times[3], 0.3 + alpha_f * 0.1, 1e-14);
        EXPECT_NEAR(run->stage_p[2], P, 1e-12);
        EXPECT_NEAR(run->stage_p[3], P + Delta, 1e-12);
    }
}

TEST(TimeLoopEndpointPressure, StagePressurePulseKeepsItsTimeIntegral)
{
    // A stage pulse of height P for one step gives endpoint values
    // 1.5P, -0.75P, 0.375P, ...; their sum is P because alpha_f (1 + rho_inf) = 1,
    // so a sensor impulse integrated over endpoints equals the stage impulse.
    constexpr double P = 3.0;
    const auto pulse = [](double t) { return (t > 0.35 && t < 0.45) ? P : 0.0; };
    const auto run = runStagePressure(0.5, 0.1, 6.0, 0.0, pulse);
    ASSERT_TRUE(run.has_value());
    ASSERT_GE(run->endpoint_p.size(), 50u);
    EXPECT_NEAR(run->endpoint_p[3], 1.5 * P, 1e-12);
    EXPECT_NEAR(run->endpoint_p[4], -0.75 * P, 1e-12);
    EXPECT_NEAR(run->endpoint_p[5], 0.375 * P, 1e-12);
    double impulse = 0.0;
    for (const double p : run->endpoint_p) {
        impulse += p * 0.1;
    }
    EXPECT_NEAR(impulse, P * 0.1, 1e-12);
}

/// du/dt + b(t, x) lambda = 0 with the constraint integral(b(t, x) u) = 0 on
/// the single tetrahedron: b = 1 before t = 0.25 and b = x after.  Returns the
/// endpoint states.
std::optional<std::vector<std::vector<Real>>> runSwitchedConstraint(double rho_inf)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto u_space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);
    auto l_space = std::make_shared<svmp::FE::spaces::L2Space>(svmp::FE::ElementType::Tetra4, 0);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_id = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 1});
    const auto l_id = sys.addField(svmp::FE::systems::FieldSpec{.name = "lambda", .space = l_space, .components = 1});
    sys.addOperator("op");
    const auto u = forms::FormExpr::trialFunction(*u_space, "u");
    const auto v = forms::FormExpr::testFunction(*u_space, "v");
    const auto l = forms::FormExpr::trialFunction(*l_space, "lambda");
    const auto q = forms::FormExpr::testFunction(*l_space, "q");
    const auto b = forms::FormExpr::coefficient(
        "active_constraint", forms::TimeScalarCoefficient([](Real x, Real, Real, Real t) {
            return t < Real(0.25) ? Real(1.0) : x;
        }));
    forms::FormCompiler compiler;
    const auto kernel = [&compiler](const forms::FormExpr& form) {
        return std::make_shared<forms::NonlinearFormKernel>(compiler.compileResidual(form), forms::ADMode::Forward);
    };
    sys.addCellKernel("op", u_id, u_id, kernel((forms::dt(u) * v).dx()));
    sys.addCellKernel("op", u_id, l_id, kernel((b * l * v).dx()));
    sys.addCellKernel("op", l_id, u_id, kernel((b * u * q).dx()));
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
    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = ts::TimeHistory::allocate(*factory, n_dofs, 2);
    // integral(u) = 0 at the start (P1 nodal values summing to zero).
    std::vector<Real> initial(static_cast<std::size_t>(n_dofs), 0.0);
    const auto u0 = static_cast<std::size_t>(sys.fieldDofOffset(u_id));
    initial[u0 + 0] = 1.0;
    initial[u0 + 1] = -2.0;
    initial[u0 + 2] = 0.5;
    initial[u0 + 3] = 0.5;
    for (int k = 1; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), initial);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(0.1);

    ts::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.6;
    opts.dt = 0.1;
    opts.max_steps = 100;
    opts.scheme = ts::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = rho_inf;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-14;
    opts.newton.rel_tolerance = 0.0;

    std::vector<std::vector<Real>> endpoints;
    ts::TimeLoopCallbacks callbacks;
    callbacks.on_step_accepted = [&endpoints, u0](ts::TimeHistory& accepted) {
        const auto span = accepted.u().localSpan();
        endpoints.emplace_back(span.begin() + static_cast<std::ptrdiff_t>(u0),
                               span.begin() + static_cast<std::ptrdiff_t>(u0 + 4u));
    };
    ts::TimeLoop loop(opts);
    const auto report = loop.run(transient, *factory, *linear, history, callbacks);
    if (!report.success) {
        ADD_FAILURE() << "TimeLoop failed: " << report.message;
        return std::nullopt;
    }
    return endpoints;
}

TEST(TimeLoopEndpointPressure, StageConstraintLeavesAnAlternatingEndpointDefectAfterAnActiveSetChange)
{
    // Diagnostic for R2.  The constraint is imposed on the stage value, so
    // after it changes from integral(u) = 0 to integral(x u) = 0 (between the
    // second and the third stage) the endpoint carries
    // B^{n+1} u_{n+1} = -rho_inf B^{n+1} u_n, which then alternates and decays
    // with -rho_inf.  With V the volume of the tetrahedron,
    // integral(x u) = V (u_0 + 2 u_1 + u_2 + u_3) / 20.
    const auto new_constraint = [](const std::vector<Real>& u) {
        return static_cast<double>(u[0] + 2.0 * u[1] + u[2] + u[3]);
    };
    for (const double rho_inf : {0.5, 0.0}) {
        SCOPED_TRACE(rho_inf);
        const auto endpoints = runSwitchedConstraint(rho_inf);
        ASSERT_TRUE(endpoints.has_value());
        ASSERT_EQ(endpoints->size(), 6u);
        const double before = new_constraint((*endpoints)[1]);
        ASSERT_GT(std::abs(before), 1e-3);
        double defect = before;
        for (std::size_t n = 2; n < endpoints->size(); ++n) {
            const double next = new_constraint((*endpoints)[n]);
            EXPECT_NEAR(next, -rho_inf * defect, 1e-12) << "step " << n;
            defect = next;
        }
    }
}

} // namespace
