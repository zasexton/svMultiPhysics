/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
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

#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <memory>
#include <vector>

namespace ts_test = svmp::FE::timestepping::test;

namespace {

std::vector<double> runReactionNorms(svmp::FE::timestepping::SchemeKind scheme,
                                     double dt,
                                     double t_end,
                                     double lambda,
                                     double generalized_alpha_rho_inf = 1.0)
{
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
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);
    history.setDt(dt);
    history.setPrevDt(dt);

    const std::vector<svmp::FE::Real> u0 = {1.0, -0.5, 0.25, 2.0};
    ts_test::setVectorByDof(history.uPrev(), u0);
    ts_test::setVectorByDof(history.uPrev2(), u0);
    {
        std::vector<svmp::FE::Real> u_dot0(u0.size(), 0.0);
        for (std::size_t i = 0; i < u0.size(); ++i) {
            u_dot0[i] = static_cast<svmp::FE::Real>(-lambda * static_cast<double>(u0[i]));
        }
        ts_test::setVectorByDof(history.uDot(), u_dot0);
    }
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = scheme;
    opts.theta = 0.5;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
	    opts.newton.residual_op = "op";
	    opts.newton.jacobian_op = "op";
	    opts.newton.max_iterations = 12;
	    // Stiff decay residuals can be large in absolute terms; use a looser absolute tolerance.
	    opts.newton.abs_tolerance = 1e-8;
	    opts.newton.rel_tolerance = 0.0;

    std::vector<double> norms;
    svmp::FE::timestepping::TimeLoopCallbacks cb;
    cb.on_step_accepted = [&norms](const svmp::FE::timestepping::TimeHistory& h) {
        norms.push_back(h.uPrev().norm());
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, cb);
    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return norms;
}

} // namespace

TEST(TimeLoopStability, StiffDecayMatchesLiteratureExpectations)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeLoopStability tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1e6;
    const double dt = 1.0;
    // Run long enough for the generalized-α stiff-mode amplification factor to
    // approach its asymptotic behavior (spectral radius at infinity), but do
    // not sample ratios at the end of the run: strongly damping schemes can
    // drive u so small that a loose absolute Newton tolerance would accept a
    // zero-update iterate (u_{n+1} ≈ u_n), corrupting the measured ratio.
    const double t_end = 6.0;

    const auto norms_be = runReactionNorms(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt, t_end, lambda);
    const auto norms_trbdf2 = runReactionNorms(svmp::FE::timestepping::SchemeKind::TRBDF2, dt, t_end, lambda);
    const auto norms_cn = runReactionNorms(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt, t_end, lambda);
    const std::vector<double> rho_values = {0.0, 0.2, 0.5, 0.9, 1.0};
    std::vector<std::vector<double>> norms_ga;
    norms_ga.reserve(rho_values.size());
    for (double rho_inf : rho_values) {
        norms_ga.push_back(runReactionNorms(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt, t_end, lambda, rho_inf));
    }

    ASSERT_GE(norms_be.size(), 2u);
    ASSERT_GE(norms_trbdf2.size(), 2u);
    ASSERT_GE(norms_cn.size(), 2u);
    for (std::size_t i = 0; i < norms_ga.size(); ++i) {
        ASSERT_GE(norms_ga[i].size(), 2u) << "rho_inf=" << rho_values[i];
    }

    auto ratioAt = [](const std::vector<double>& norms, std::size_t k) -> double {
        EXPECT_GT(norms.size(), k);
        if (norms.size() <= k || norms[k - 1] == 0.0) {
            return 0.0;
        }
        return norms[k] / norms[k - 1];
    };

    // Use early ratios for stiffly-accurate schemes to avoid tolerance-induced plateaus.
    const double ratio_be = ratioAt(norms_be, 2);
    const double ratio_trbdf2 = ratioAt(norms_trbdf2, 2);
    const double ratio_cn = ratioAt(norms_cn, 2);

    // Backward Euler and TRBDF2 are (stiffly) strongly damping for stiff decay.
    EXPECT_LT(ratio_be, 1e-3);
    EXPECT_LT(ratio_trbdf2, 1e-2);

    // Crank–Nicolson (theta=0.5) is A-stable but not L-stable; stiff modes are not damped.
    EXPECT_GT(ratio_cn, 0.9);

    // Generalized-α controls high-frequency damping via ρ∞; decreasing ρ∞ increases damping.
    const std::size_t sample_idx = 4;
    std::vector<double> ratios_ga;
    ratios_ga.reserve(norms_ga.size());
    for (const auto& norms : norms_ga) {
        ASSERT_GT(norms.size(), sample_idx);
        ratios_ga.push_back(ratioAt(norms, sample_idx));
    }
    for (std::size_t i = 1; i < ratios_ga.size(); ++i) {
        EXPECT_LT(ratios_ga[i - 1], ratios_ga[i]) << "rho_inf=" << rho_values[i - 1] << " vs " << rho_values[i];
    }
    for (std::size_t i = 0; i < ratios_ga.size(); ++i) {
        EXPECT_NEAR(ratios_ga[i], rho_values[i], 0.12) << "rho_inf=" << rho_values[i];
    }
}
