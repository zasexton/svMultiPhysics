/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
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
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace svmp::FE::timestepping::test {

namespace {

using svmp::FE::Real;

// Run a free vibration problem M a + K u = 0 (no physical damping)
// and measure the algorithmic damping.
double measureAmplificationFactor(SchemeKind scheme,
                                  double dt,
                                  double omega,
                                  double rho_inf = 1.0)
{
    // M = 1, K = omega^2.
    // u'' + omega^2 u = 0.
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();

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
    auto linear = factory->createLinearSolver(directSolve());

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 1.0, 1.0, 1.0}; // Uniform initial displacement
    setVectorByDof(history.uPrev(), u0);

    // Initial velocity = 0.
    const std::vector<Real> v0(u0.size(), 0.0);
    setVectorByDof(history.uDot(), v0);

    // Initial acceleration = -omega^2 * u0.
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDDot(), a0);

    // Prime u^{n-1} consistent with exact solution u(t) = cos(omega*t) at t=-dt.
    const double c = std::cos(omega * dt);
    // const double s = std::sin(omega * dt);
    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * c);
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);

    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.dt = dt;
    opts.max_steps = 10;
    opts.t_end = dt * opts.max_steps;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = rho_inf;
    // For Newmark, we map rho_inf to beta/gamma if needed, or just use defaults.
    // But here we'll use GeneralizedAlpha for the main test.
    if (scheme == SchemeKind::Newmark) {
        opts.newmark_beta = 0.25;
        opts.newmark_gamma = 0.5;
    }
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 25;
    // Residual norms scale with omega^2. Use a looser absolute tolerance so
    // stiff/high-frequency oscillator problems don't stall at ~1e-11.
    opts.newton.abs_tolerance = 1e-8;
    opts.newton.rel_tolerance = 0.0;

    // Run for enough steps to estimate growth/decay.
    // We'll just run 1 step and compare energy or amplitude?
    // Actually, spectral radius is asymptotic.
    // Let's run for a few periods.
    // But for high frequency (omega*dt >> 1), we want to see immediate damping.

    // Reset history
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);
    setVectorByDof(history.uPrev2(), u_minus_dt);
    history.setTime(0.0);
    history.resetCurrentToPrevious();
    
    opts.max_steps = 20;
    opts.t_end = dt * 20;
    TimeLoop loop20(opts);
    const auto rep = loop20.run(transient, *factory, *linear, history);
    
    if (!rep.success) return 1e9;
    
    const double u_final = getVectorByDof(history.uPrev())[0]; // All DOFs same
    const double u_init = u0[0];
    
    // Effective growth per step = (u_final / u_init)^(1/steps)
    // This is rough because phase varies.
    // Better: look at total energy E = 0.5*v^2 + 0.5*omega^2*u^2.
    
    const double v_final = getVectorByDof(history.uDot())[0];
    const double E_final = 0.5 * v_final * v_final + 0.5 * omega * omega * u_final * u_final;
    
    const double v_init = v0[0]; // 0
    const double E_init = 0.5 * v_init * v_init + 0.5 * omega * omega * u_init * u_init;
    
    return std::sqrt(E_final / E_init); // Amplitude ratio
}

std::vector<double> runFirstOrderOscillatorEnergies(SchemeKind scheme,
                                                    double dt,
                                                    double t_end,
                                                    double omega)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto q_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "q", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto q = svmp::FE::forms::FormExpr::trialFunction(*space, "q");
    const auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    const auto vq = svmp::FE::forms::FormExpr::testFunction(*space, "vq");
    const auto vp = svmp::FE::forms::FormExpr::testFunction(*space, "vp");

    // Hamiltonian oscillator:
    //   q' - p = 0
    //   p' + omega^2 q = 0
    const auto form_qq = (svmp::FE::forms::dt(q) * vq).dx();
    const auto form_qp = (-(p * vq)).dx();
    const auto form_pp = (svmp::FE::forms::dt(p) * vp).dx();
    const auto form_pq = ((q * vp) * static_cast<Real>(omega * omega)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir_qq = compiler.compileResidual(form_qq);
    auto ir_qp = compiler.compileResidual(form_qp);
    auto ir_pp = compiler.compileResidual(form_pp);
    auto ir_pq = compiler.compileResidual(form_pq);

    auto k_qq = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_qq), svmp::FE::forms::ADMode::Forward);
    auto k_qp = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_qp), svmp::FE::forms::ADMode::Forward);
    auto k_pp = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_pp), svmp::FE::forms::ADMode::Forward);
    auto k_pq = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_pq), svmp::FE::forms::ADMode::Forward);

    sys.addCellKernel("op", q_field, q_field, k_qq);
    sys.addCellKernel("op", q_field, p_field, k_qp);
    sys.addCellKernel("op", p_field, p_field, k_pp);
    sys.addCellKernel("op", p_field, q_field, k_pq);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    auto linear = factory->createLinearSolver(directSolve());

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/2);
    history.setPrevDt(dt);

    const auto n_q = sys.fieldDofHandler(q_field).getNumDofs();
    const auto n_p = sys.fieldDofHandler(p_field).getNumDofs();
    if (n_q != n_p || n_q <= 0) {
        ADD_FAILURE() << "Unexpected per-field DOF counts";
        return {};
    }
    const auto off_q = sys.fieldDofOffset(q_field);
    const auto off_p = sys.fieldDofOffset(p_field);
    if (off_q < 0 || off_p < 0) {
        ADD_FAILURE() << "Invalid field offsets";
        return {};
    }

    std::vector<Real> u0(static_cast<std::size_t>(n_dofs), 0.0);
    for (svmp::FE::GlobalIndex i = 0; i < n_q; ++i) {
        u0[static_cast<std::size_t>(off_q + i)] = static_cast<Real>(1.0);
        u0[static_cast<std::size_t>(off_p + i)] = static_cast<Real>(0.0);
    }
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = static_cast<int>(std::ceil(t_end / dt)) + 2;
    opts.scheme = scheme;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    std::vector<double> energies;
    TimeLoopCallbacks cb;
    cb.on_step_accepted = [&](const svmp::FE::timestepping::TimeHistory& h) {
        const auto s = h.uPrevSpan();
        double E = 0.0;
        for (svmp::FE::GlobalIndex i = 0; i < n_q; ++i) {
            const double qi = static_cast<double>(s[static_cast<std::size_t>(off_q + i)]);
            const double pi = static_cast<double>(s[static_cast<std::size_t>(off_p + i)]);
            E += 0.5 * (pi * pi + (omega * omega) * qi * qi);
        }
        energies.push_back(E);
    };

    TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, cb);
    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);
    return energies;
}

} // namespace

TEST(TimeLoopDissipation, GeneralizedAlpha_DampsHighFrequencies)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    // High frequency: omega * dt >> 1
    const double dt = 0.1;
    const double omega = 1000.0; // omega*dt = 100
    
    // rho_inf = 1.0 (No dissipation)
    // Expect Energy ratio ~ 1.0 (maybe slightly less due to numerics, but close)
    double amp_1 = measureAmplificationFactor(SchemeKind::GeneralizedAlpha, dt, omega, 1.0);
    EXPECT_NEAR(amp_1, 1.0, 0.05);

    // rho_inf = 0.5
    // Expect significant damping. Spectral radius is 0.5 at infinity.
    // After 20 steps, amplitude should drop significantly.
    // Actually, the "Amplitude ratio" returned is E_final/E_init (sqrt).
    // If rho(infinity) = 0.5, then per step contraction is ~0.5.
    // Over 20 steps, it should be tiny.
    double amp_05 = measureAmplificationFactor(SchemeKind::GeneralizedAlpha, dt, omega, 0.5);
    EXPECT_LT(amp_05, 0.1); 
    
    // rho_inf = 0.0 (Asymptotic Annihilation)
    double amp_00 = measureAmplificationFactor(SchemeKind::GeneralizedAlpha, dt, omega, 0.0);
    EXPECT_LT(amp_00, 1e-6);
}

TEST(TimeLoopDissipation, Newmark_PreservesEnergyInTrapezoidalLimit)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    // Trapezoidal rule (Average acceleration): beta=0.25, gamma=0.5.
    // Should preserve energy for linear oscillator.
    
    const double dt = 0.1;
    const double omega = 10.0; // omega*dt = 1.0 (resolved)
    
    double amp = measureAmplificationFactor(SchemeKind::Newmark, dt, omega); // Uses defaults beta=0.25, gamma=0.5
    EXPECT_NEAR(amp, 1.0, 1e-6);
}

TEST(TimeLoopDissipation, CG2_PreservesEnergyOnFirstOrderOscillator)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double pi = std::acos(-1.0);
    const double omega = 2.0 * pi;
    const double dt = 0.2;
    const double t_end = 20.0; // 20 periods

    const auto energies = runFirstOrderOscillatorEnergies(SchemeKind::CG2, dt, t_end, omega);
    ASSERT_FALSE(energies.empty());
    const double E0 = energies.front();
    ASSERT_GT(E0, 0.0);

    double max_rel = 0.0;
    for (double E : energies) {
        max_rel = std::max(max_rel, std::abs(E - E0) / E0);
    }
    EXPECT_LT(max_rel, 1e-6);
}

} // namespace svmp::FE::timestepping::test
