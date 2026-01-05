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
#include "Backends/Utils/BackendOptions.h"

#include "Core/Types.h"

#include "Forms/Forms.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"

#include "Spaces/H1Space.h"

#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"
#include "TimeStepping/VSVO_BDF_Controller.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace ts_test = svmp::FE::timestepping::test;

namespace {

using ts_test::createTestFactory;
using ts_test::directSolve;
using ts_test::getVectorByDof;
using ts_test::relativeL2Error;
using ts_test::setVectorByDof;
using ts_test::singleTetraTopology;

[[nodiscard]] svmp::FE::dofs::MeshTopologyInfo twoTetraSharedFaceTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 5;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 1, 2, 3, 4};
    topo.vertex_gids = {0, 1, 2, 3, 4};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

[[nodiscard]] std::pair<double, double> coupledReactionExact2x2(double t,
                                                                double u0,
                                                                double w0,
                                                                double lambda_u,
                                                                double lambda_w,
                                                                double kappa)
{
    // System: u' + lambda_u*u + kappa*w = 0,  w' + kappa*u + lambda_w*w = 0.
    // Matrix form: x' = A x,  A = [[-lambda_u, -kappa], [-kappa, -lambda_w]].
    const double a = -lambda_u;
    const double d = -lambda_w;
    const double b = -kappa;

    const double m = 0.5 * (a + d);
    const double half_diff = 0.5 * (a - d);
    const double s = std::sqrt(half_diff * half_diff + b * b);

    const double exp_mt = std::exp(m * t);
    if (s == 0.0) {
        return {exp_mt * u0, exp_mt * w0};
    }

    const double cosh_st = std::cosh(s * t);
    const double sinh_over_s = std::sinh(s * t) / s;

    const double a_m = a - m;
    const double d_m = d - m;

    const double u = exp_mt * (cosh_st * u0 + sinh_over_s * (a_m * u0 + b * w0));
    const double w = exp_mt * (cosh_st * w0 + sinh_over_s * (b * u0 + d_m * w0));
    return {u, w};
}

std::vector<Real> runReactionProblem(svmp::FE::timestepping::SchemeKind scheme,
                                     double dt,
                                     double t_end,
                                     double lambda,
                                     int history_depth = 2,
                                     std::shared_ptr<svmp::FE::timestepping::StepController> controller = {},
                                     double generalized_alpha_rho_inf = 1.0,
                                     int dg_degree = 1,
                                     int cg_degree = 2,
                                     svmp::FE::timestepping::CollocationSolveStrategy collocation_solve =
                                         svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                     int collocation_max_outer_iterations = 4,
                                     double collocation_outer_tolerance = 0.0,
                                     bool exact_initial_history = false,
                                     double theta = 0.5)
{
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
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);
    if (!sys.isSetup()) {
        ADD_FAILURE() << "FESystem::setup did not mark the system as setup";
        return {};
    }

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

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, history_depth);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    if (u0.size() != static_cast<std::size_t>(n_dofs)) {
        ADD_FAILURE() << "Unexpected DOF count: got " << n_dofs << ", expected " << u0.size();
        return {};
    }
    for (int k = 1; k <= history.historyDepth(); ++k) {
        if (exact_initial_history) {
            const double t_k = -static_cast<double>(k - 1) * dt;
            const double scale = std::exp(-lambda * t_k);
            std::vector<Real> u_k(u0.size(), 0.0);
            for (std::size_t i = 0; i < u0.size(); ++i) {
                u_k[i] = static_cast<Real>(static_cast<double>(u0[i]) * scale);
            }
            ts_test::setVectorByDof(history.uPrevK(k), u_k);
        } else {
            ts_test::setVectorByDof(history.uPrevK(k), u0);
        }
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = scheme;
    opts.theta = theta;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.dg_degree = dg_degree;
    opts.cg_degree = cg_degree;
    opts.collocation_solve = collocation_solve;
    opts.collocation_max_outer_iterations = collocation_max_outer_iterations;
    opts.collocation_outer_tolerance = collocation_outer_tolerance;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    opts.step_controller = std::move(controller);

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm
                      << " step=" << history.stepIndex()
                      << " t=" << history.time()
                      << " dt=" << history.dt()
                      << " dt_prev=" << history.dtPrev() << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return ts_test::getVectorByDof(history.uPrev());
}

std::vector<Real> runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind scheme,
                                                    double dt,
                                                    double t_end,
                                                    double lambda_u,
                                                    double lambda_w,
                                                    double kappa,
                                                    double theta = 0.5,
                                                    int history_depth = 2,
                                                    bool exact_initial_history = false)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoTetraSharedFaceMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto w_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "w", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto w = svmp::FE::forms::FormExpr::trialFunction(*space, "w");
    const auto v_u = svmp::FE::forms::FormExpr::testFunction(*space, "v_u");
    const auto v_w = svmp::FE::forms::FormExpr::testFunction(*space, "v_w");

    const auto form_uu = (svmp::FE::forms::dt(u) * v_u + (u * v_u) * static_cast<Real>(lambda_u)).dx();
    const auto form_uw = ((w * v_u) * static_cast<Real>(kappa)).dx();
    const auto form_wu = ((u * v_w) * static_cast<Real>(kappa)).dx();
    const auto form_ww = (svmp::FE::forms::dt(w) * v_w + (w * v_w) * static_cast<Real>(lambda_w)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir_uu = compiler.compileResidual(form_uu);
    auto ir_uw = compiler.compileResidual(form_uw);
    auto ir_wu = compiler.compileResidual(form_wu);
    auto ir_ww = compiler.compileResidual(form_ww);

    auto kernel_uu = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_uu), svmp::FE::forms::ADMode::Forward);
    auto kernel_uw = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_uw), svmp::FE::forms::ADMode::Forward);
    auto kernel_wu = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_wu), svmp::FE::forms::ADMode::Forward);
    auto kernel_ww = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_ww), svmp::FE::forms::ADMode::Forward);

    sys.addCellKernel("op", u_field, u_field, kernel_uu);
    sys.addCellKernel("op", u_field, w_field, kernel_uw);
    sys.addCellKernel("op", w_field, u_field, kernel_wu);
    sys.addCellKernel("op", w_field, w_field, kernel_ww);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = twoTetraSharedFaceTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, history_depth);

    const auto n_u = sys.fieldDofHandler(u_field).getNumDofs();
    const auto n_w = sys.fieldDofHandler(w_field).getNumDofs();
    if (n_u != n_w || n_u <= 0) {
        ADD_FAILURE() << "Unexpected field DOF counts";
        return {};
    }
    const auto offset_u = sys.fieldDofOffset(u_field);
    const auto offset_w = sys.fieldDofOffset(w_field);
    if (offset_u < 0 || offset_w < 0) {
        ADD_FAILURE() << "Invalid field offsets";
        return {};
    }

    // TwoTetraSharedFaceMeshAccess with H1(Tetra4,P1) yields 5 vertex DOFs per field.
    if (n_u != 5) {
        ADD_FAILURE() << "Unexpected per-field DOF count: got " << n_u << ", expected 5";
        return {};
    }
    const std::vector<double> u0 = {1.0, -0.5, 0.25, 2.0, -1.5};
    const std::vector<double> w0 = {0.5, -0.25, 0.75, -1.0, 0.1};

    auto exactAt = [&](double t) -> std::vector<Real> {
        std::vector<Real> out(static_cast<std::size_t>(n_dofs), 0.0);
        for (GlobalIndex i = 0; i < n_u; ++i) {
            const auto [ui, wi] = coupledReactionExact2x2(t, u0[static_cast<std::size_t>(i)],
                                                          w0[static_cast<std::size_t>(i)],
                                                          lambda_u, lambda_w, kappa);
            out[static_cast<std::size_t>(offset_u + i)] = static_cast<Real>(ui);
            out[static_cast<std::size_t>(offset_w + i)] = static_cast<Real>(wi);
        }
        return out;
    };

    for (int k = 1; k <= history.historyDepth(); ++k) {
        const double t_k = exact_initial_history ? -static_cast<double>(k - 1) * dt : 0.0;
        setVectorByDof(history.uPrevK(k), exactAt(t_k));
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 10000;
    opts.scheme = scheme;
    opts.theta = theta;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&,
                                             const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm
                      << " step=" << history.stepIndex()
                      << " t=" << history.time()
                      << " dt=" << history.dt()
                      << " dt_prev=" << history.dtPrev() << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::shared_ptr<svmp::FE::timestepping::VSVO_BDF_Controller>
makeFixedVsvoBdfController(int order, double dt)
{
    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    // For fixed-step/order convergence tests, keep the controller inert.
    ctrl_opts.abs_tol = 1.0;
    ctrl_opts.rel_tol = 0.0;
    ctrl_opts.min_order = order;
    ctrl_opts.max_order = order;
    ctrl_opts.initial_order = order;
    ctrl_opts.max_retries = 0;
    ctrl_opts.safety = 1.0;
    ctrl_opts.min_factor = 1.0;
    ctrl_opts.max_factor = 1.0;
    ctrl_opts.min_dt = dt;
    ctrl_opts.max_dt = dt;
    ctrl_opts.pi_alpha = 0.0;
    ctrl_opts.pi_beta = 0.0;
    ctrl_opts.increase_order_threshold = 0.0;

    return std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
}

std::vector<Real> runLogisticProblem(svmp::FE::timestepping::SchemeKind scheme,
                                     double dt,
                                     double t_end,
                                     double r,
                                     double K = 1.0)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const Real rr = static_cast<Real>(r);
    const Real inv_K = static_cast<Real>(1.0 / K);

    // Logistic: u' = r*u*(1 - u/K) = r*u - r/K*u^2
    // Residual: u' - r*u + r/K*u^2 = 0.
    const auto form =
        (svmp::FE::forms::dt(u) * v - (u * v) * rr + (u * u * v) * (rr * inv_K)).dx();

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
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs);

    const std::vector<Real> u0 = {0.2, 0.4, 0.6, 0.8};
    if (u0.size() != static_cast<std::size_t>(n_dofs)) {
        ADD_FAILURE() << "Unexpected DOF count: got " << n_dofs << ", expected " << u0.size();
        return {};
    }
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 2000;
    opts.scheme = scheme;
    opts.theta = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runHeatManufacturedSinForcing(double dt, double t_end)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    double current_time = 0.0;
    const auto f = svmp::FE::forms::FormExpr::coefficient(
        "f",
        [&current_time](Real, Real, Real) { return static_cast<Real>(std::sin(current_time)); });

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form = (svmp::FE::forms::dt(u) * v +
                       svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)) -
                       f * v)
                          .dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());

    const std::vector<Real> u0(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_step_start = [&current_time](const svmp::FE::timestepping::TimeHistory& h) {
        current_time = h.time() + h.dt();
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runOscillatorDt2(double dt, double t_end, double omega)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * c);
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind scheme,
                                             double dt,
                                             double t_end,
                                             double omega,
                                             double generalized_alpha_rho_inf = 1.0)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * c);
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);

    const std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 2000;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.newmark_beta = 0.25;
    opts.newmark_gamma = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

struct ScalarTrajectory {
    std::vector<double> times;
    std::vector<double> values;
};

std::optional<double> estimatePeriodFromDownwardZeroCrossings(const ScalarTrajectory& traj)
{
    if (traj.times.size() != traj.values.size()) {
        return std::nullopt;
    }
    if (traj.times.size() < 2) {
        return std::nullopt;
    }

    std::vector<double> crossing_times;
    crossing_times.reserve(traj.times.size() / 2);

    for (std::size_t i = 1; i < traj.values.size(); ++i) {
        const double y0 = traj.values[i - 1];
        const double y1 = traj.values[i];
        if (!(y0 > 0.0 && y1 <= 0.0)) {
            continue;
        }
        const double t0 = traj.times[i - 1];
        const double t1 = traj.times[i];
        const double denom = y1 - y0;
        if (denom == 0.0) {
            continue;
        }
        const double s = (0.0 - y0) / denom;
        crossing_times.push_back(t0 + s * (t1 - t0));
    }

    if (crossing_times.size() < 3) {
        return std::nullopt;
    }

    double sum = 0.0;
    int count = 0;
    for (std::size_t i = 1; i < crossing_times.size(); ++i) {
        const double dt = crossing_times[i] - crossing_times[i - 1];
        if (dt > 0.0 && std::isfinite(dt)) {
            sum += dt;
            count += 1;
        }
    }
    if (count == 0) {
        return std::nullopt;
    }
    return sum / static_cast<double>(count);
}

ScalarTrajectory runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind scheme,
                                                      double dt,
                                                      double t_end,
                                                      double omega,
                                                      double generalized_alpha_rho_inf = 1.0)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * c);
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);

    const std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = static_cast<int>(std::ceil(t_end / dt)) + 10;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.newmark_beta = 0.25;
    opts.newmark_gamma = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    ScalarTrajectory traj;
    traj.times.push_back(0.0);
    traj.values.push_back(static_cast<double>(u0[0]));

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    svmp::FE::timestepping::NewtonReport last_nr;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&,
                                             const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };
    callbacks.on_step_accepted = [&traj](const svmp::FE::timestepping::TimeHistory& h) {
        const auto up = h.uPrevSpan();
        if (up.empty()) {
            return;
        }
        traj.times.push_back(h.time());
        traj.values.push_back(static_cast<double>(up[0]));
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm
                      << " step=" << history.stepIndex()
                      << " t=" << history.time()
                      << " dt=" << history.dt()
                      << " dt_prev=" << history.dtPrev() << ")";
        return traj;
    }
    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return traj;
}

std::vector<Real> runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind scheme,
                                                   double dt,
                                                   double t_end,
                                                   double omega,
                                                   double zeta,
                                                   double generalized_alpha_rho_inf = 1.0)
{
    if (!(zeta >= 0.0 && zeta < 1.0)) {
        ADD_FAILURE() << "runDampedOscillatorDt2Structural: zeta must be in [0,1)";
        return {};
    }

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v +
         svmp::FE::forms::dt(u) * v * static_cast<Real>(2.0 * zeta * omega) +
         (u * v) * static_cast<Real>(omega * omega))
            .dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    // Prime u^{n-1} using the exact damped oscillator at t=-dt with v(0)=0.
    const double omega_d = omega * std::sqrt(1.0 - zeta * zeta);
    const double alpha = zeta / std::sqrt(1.0 - zeta * zeta);
    const double c = std::cos(omega_d * dt);
    const double s = std::sin(omega_d * dt);
    const double expm = std::exp(zeta * omega * dt);
    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * expm * (c - alpha * s));
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);

    const std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 4000;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.newmark_beta = 0.25;
    opts.newmark_gamma = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 16;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind scheme,
                                              double dt,
                                              double t_end,
                                              double omega,
                                              int dg_degree = 1,
                                              int cg_degree = 2)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    // Choose v0 = omega*u0 so the exact solution is u(t)=u0*(cos(omega t)+sin(omega t)).
    std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        v0[i] = static_cast<Real>(omega * static_cast<double>(u0[i]));
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    const double s = std::sin(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c - s));
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 2000;
    opts.scheme = scheme;
    opts.dg_degree = dg_degree;
    opts.cg_degree = cg_degree;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

} // namespace

TEST(NewtonSolverSanity, LinearReactionConvergesInOneUpdate)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    constexpr double dt = 0.2;
    constexpr double lambda = 1.0;

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
    history.setDt(dt);
    history.setPrevDt(dt);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    // Workspace.
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);
    ASSERT_TRUE(ws.isAllocated());

    auto& J = *ws.jacobian;
    auto& r = *ws.residual;
    auto& du = *ws.delta;

    // Assemble at u=u_prev.
    svmp::FE::systems::SystemStateView state;
    state.time = dt;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = history.uSpan();
    state.u_prev = history.uPrevSpan();
    state.u_prev2 = history.uPrev2Span();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    auto J_view = J.createAssemblyView();
    auto r_view = r.createAssemblyView();
    ASSERT_NE(J_view.get(), nullptr);
    ASSERT_NE(r_view.get(), nullptr);
    (void)transient.assemble(req, state, J_view.get(), r_view.get());
    const double r0 = r.norm();
    ASSERT_GT(r0, 0.0);

    du.zero();
    const auto linrep = linear->solve(J, du, r);
    ASSERT_TRUE(linrep.converged);

    // u <- u - du
    {
        auto us = history.u().localSpan();
        auto dus = du.localSpan();
        ASSERT_EQ(us.size(), dus.size());
        for (std::size_t i = 0; i < us.size(); ++i) {
            us[i] -= dus[i];
        }
    }

    // Reassemble residual at updated u.
    auto r_view2 = r.createAssemblyView();
    ASSERT_NE(r_view2.get(), nullptr);
    req.want_matrix = false;
    req.want_vector = true;
    (void)transient.assemble(req, state, nullptr, r_view2.get());
    const double r1 = r.norm();
    EXPECT_LT(r1, 1e-12);
}

TEST(NewtonSolverSanity, SolveStepConvergesForLinearReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    constexpr double dt = 0.2;
    constexpr double lambda = 1.0;

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
    history.setDt(dt);
    history.setPrevDt(dt);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
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

    const auto rep = newton.solveStep(transient, *linear, dt, history, ws);
    EXPECT_TRUE(rep.converged);
    EXPECT_LE(rep.iterations, 2);
}

TEST(TimeLoopConvergence, BackwardEuler_IsFirstOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale1 = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale1));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopSanity, BackwardEuler_SingleStep_AdvancesSolution)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double dt = 0.2;

    const auto u_dt = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt, dt, lambda);
    ASSERT_EQ(u_dt.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const double scale = 1.0 / (1.0 + lambda * dt);
    for (std::size_t i = 0; i < u_dt.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(u_dt[i]), static_cast<double>(u0[i]) * scale, 1e-12);
    }
}

TEST(TimeLoopConvergence, Bdf2_IsSecondOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BDF2, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BDF2, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, CoupledTwoFieldReaction_ThetaMethod_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda_u = 1.0;
    const double lambda_w = 2.0;
    const double kappa = 0.5;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto x_dt1 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod,
                                                         dt1,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/false);
    const auto x_dt2 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod,
                                                         dt2,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/false);
    ASSERT_EQ(x_dt1.size(), 10u);
    ASSERT_EQ(x_dt2.size(), 10u);

    const std::vector<double> u0 = {1.0, -0.5, 0.25, 2.0, -1.5};
    const std::vector<double> w0 = {0.5, -0.25, 0.75, -1.0, 0.1};
    std::vector<Real> exact(10u, 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        const auto [ui, wi] = coupledReactionExact2x2(t_end, u0[i], w0[i], lambda_u, lambda_w, kappa);
        exact[i] = static_cast<Real>(ui);
        exact[5u + i] = static_cast<Real>(wi);
    }

    const double e1 = relativeL2Error(x_dt1, exact);
    const double e2 = relativeL2Error(x_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, CoupledTwoFieldReaction_Bdf2_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda_u = 1.0;
    const double lambda_w = 2.0;
    const double kappa = 0.5;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto x_dt1 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::BDF2,
                                                         dt1,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/true);
    const auto x_dt2 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::BDF2,
                                                         dt2,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/true);
    ASSERT_EQ(x_dt1.size(), 10u);
    ASSERT_EQ(x_dt2.size(), 10u);

    const std::vector<double> u0 = {1.0, -0.5, 0.25, 2.0, -1.5};
    const std::vector<double> w0 = {0.5, -0.25, 0.75, -1.0, 0.1};
    std::vector<Real> exact(10u, 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        const auto [ui, wi] = coupledReactionExact2x2(t_end, u0[i], w0[i], lambda_u, lambda_w, kappa);
        exact[i] = static_cast<Real>(ui);
        exact[5u + i] = static_cast<Real>(wi);
    }

    const double e1 = relativeL2Error(x_dt1, exact);
    const double e2 = relativeL2Error(x_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, VSVO_BDF_FixedOrder3_IsThirdOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const int order = 3;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt1, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt1),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt2, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt2),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 2.4);
}

TEST(TimeLoopConvergence, VSVO_BDF_FixedOrder4_IsFourthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const int order = 4;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt1, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt1),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt2, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt2),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 3.2);
}

TEST(TimeLoopConvergence, VSVO_BDF_FixedOrder5_IsFifthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const int order = 5;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt1, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt1),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt2, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt2),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 4.0);
}

TEST(TimeLoopConvergence, ThetaMethod_CrankNicolson_IsSecondOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, ThetaMethod_Theta075_IsFirstOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const double theta = 0.75;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/false,
                                          /*theta=*/theta);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/false,
                                          /*theta=*/theta);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopConvergence, ThetaMethod_CrankNicolson_IsSecondOrder_ForLogisticEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double r = 1.0;
    const double K = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;
    const double dt3 = 0.05;

    const auto u_dt1 = runLogisticProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt1, t_end, r, K);
    const auto u_dt2 = runLogisticProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt2, t_end, r, K);
    const auto u_dt3 = runLogisticProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt3, t_end, r, K);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);
    ASSERT_EQ(u_dt3.size(), 4u);

    auto l2Diff = [](const std::vector<Real>& a, const std::vector<Real>& b) -> double {
        EXPECT_EQ(a.size(), b.size());
        double sum = 0.0;
        const std::size_t n = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i] - b[i]);
            sum += d * d;
        }
        return std::sqrt(sum);
    };

    // Self-convergence against successive refinements avoids needing an analytic
    // solution for the coupled FE semi-discretization.
    const double e12 = l2Diff(u_dt1, u_dt2);
    const double e23 = l2Diff(u_dt2, u_dt3);
    const double p = std::log(e12 / e23) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e12=" << e12 << " e23=" << e23;
}

TEST(TimeLoopConvergence, Trbdf2_IsSecondOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::TRBDF2, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::TRBDF2, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, DG1_IsThirdOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG1, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG1, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 2.6);
}

TEST(TimeLoopConvergence, DG_Degree2_IsFifthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/2);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/2);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 4.0);
}

TEST(TimeLoopConvergence, DG_Degree3_IsSeventhOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/3);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/3);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 6.0) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, CG2_IsFourthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG2, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG2, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 3.2);
}

TEST(TimeLoopConvergence, CG_Degree3_IsSixthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/3);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/3);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 5.0);
}

TEST(TimeLoopConvergence, CG_Degree4_IsEighthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.5;
    const double dt2 = 0.25;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/4);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/4);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 7.0) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopVerification, ManufacturedHeatSinForcing_ConvergesWithBackwardEuler)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runHeatManufacturedSinForcing(dt1, t_end);
    const auto u_dt2 = runHeatManufacturedSinForcing(dt2, t_end);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const Real exact_value = static_cast<Real>(1.0 - std::cos(t_end));
    std::vector<Real> exact(u_dt1.size(), exact_value);

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopVerification, Dt2Oscillator_BackwardDifference_IsFirstOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2(dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2(dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopConvergence, Dt2Oscillator_Newmark_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, Dt2Oscillator_Newmark_LongTimePeriod_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double pi = std::acos(-1.0);
    const double omega = 2.0 * pi;
    const double exact_period = 2.0 * pi / omega;
    const int cycles = 20;
    const double t_end = static_cast<double>(cycles) * exact_period;

    const double dt1 = 0.05;
    const double dt2 = 0.025;

    const auto traj1 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::Newmark, dt1, t_end, omega);
    const auto traj2 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::Newmark, dt2, t_end, omega);

    const auto T1 = estimatePeriodFromDownwardZeroCrossings(traj1);
    const auto T2 = estimatePeriodFromDownwardZeroCrossings(traj2);
    ASSERT_TRUE(T1.has_value());
    ASSERT_TRUE(T2.has_value());

    const double e1 = std::abs(*T1 - exact_period);
    const double e2 = std::abs(*T2 - exact_period);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "T1=" << *T1 << " T2=" << *T2 << " e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2DampedOscillator_Newmark_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double zeta = 0.1;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt1, t_end, omega, zeta);
    const auto u_dt2 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt2, t_end, omega, zeta);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double omega_d = omega * std::sqrt(1.0 - zeta * zeta);
    const double alpha = zeta / std::sqrt(1.0 - zeta * zeta);
    const double exp_decay = std::exp(-zeta * omega * t_end);
    const double c = std::cos(omega_d * t_end);
    const double s = std::sin(omega_d * t_end);

    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * exp_decay * (c + alpha * s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2Oscillator_GeneralizedAlphaSecondOrder_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, omega,
                                                  /*generalized_alpha_rho_inf=*/1.0);
    const auto u_dt2 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, omega,
                                                  /*generalized_alpha_rho_inf=*/1.0);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, Dt2Oscillator_GeneralizedAlphaSecondOrder_LongTimePeriod_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double pi = std::acos(-1.0);
    const double omega = 2.0 * pi;
    const double exact_period = 2.0 * pi / omega;
    const int cycles = 20;
    const double t_end = static_cast<double>(cycles) * exact_period;

    const double dt1 = 0.05;
    const double dt2 = 0.025;

    const auto traj1 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                            dt1,
                                                            t_end,
                                                            omega,
                                                            /*generalized_alpha_rho_inf=*/1.0);
    const auto traj2 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                            dt2,
                                                            t_end,
                                                            omega,
                                                            /*generalized_alpha_rho_inf=*/1.0);

    const auto T1 = estimatePeriodFromDownwardZeroCrossings(traj1);
    const auto T2 = estimatePeriodFromDownwardZeroCrossings(traj2);
    ASSERT_TRUE(T1.has_value());
    ASSERT_TRUE(T2.has_value());

    const double e1 = std::abs(*T1 - exact_period);
    const double e2 = std::abs(*T2 - exact_period);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "T1=" << *T1 << " T2=" << *T2 << " e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2DampedOscillator_GeneralizedAlphaSecondOrder_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double zeta = 0.1;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                        dt1,
                                                        t_end,
                                                        omega,
                                                        zeta,
                                                        /*generalized_alpha_rho_inf=*/1.0);
    const auto u_dt2 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                        dt2,
                                                        t_end,
                                                        omega,
                                                        zeta,
                                                        /*generalized_alpha_rho_inf=*/1.0);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double omega_d = omega * std::sqrt(1.0 - zeta * zeta);
    const double alpha = zeta / std::sqrt(1.0 - zeta * zeta);
    const double exp_decay = std::exp(-zeta * omega * t_end);
    const double c = std::cos(omega_d * t_end);
    const double s = std::sin(omega_d * t_end);

    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * exp_decay * (c + alpha * s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2Oscillator_GeneralizedAlphaSecondOrder_RhoInfSweep_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const std::vector<double> rho_values = {0.0, 0.2, 0.5, 0.9, 1.0};
    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    for (double rho_inf : rho_values) {
        const auto u_dt1 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, omega,
                                                      /*generalized_alpha_rho_inf=*/rho_inf);
        const auto u_dt2 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, omega,
                                                      /*generalized_alpha_rho_inf=*/rho_inf);
        ASSERT_EQ(u_dt1.size(), 4u);
        ASSERT_EQ(u_dt2.size(), 4u);

        const double e1 = relativeL2Error(u_dt1, exact);
        const double e2 = relativeL2Error(u_dt2, exact);
        const double p = std::log(e1 / e2) / std::log(2.0);
        EXPECT_GT(p, 1.6) << "rho_inf=" << rho_inf << " e1=" << e1 << " e2=" << e2;
    }
}

TEST(TimeLoopConvergence, Dt2Oscillator_DG1_IsThirdOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::DG1, dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::DG1, dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double c = std::cos(omega * t_end);
    const double s = std::sin(omega * t_end);
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c + s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 2.6);
}

TEST(TimeLoopConvergence, Dt2Oscillator_CG2_IsFourthOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::CG2, dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::CG2, dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double c = std::cos(omega * t_end);
    const double s = std::sin(omega * t_end);
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c + s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 3.2);
}

TEST(TimeLoopConvergence, GeneralizedAlpha_FirstOrder_IsSecondOrderOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, lambda,
                                          /*history_depth=*/3);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, lambda,
                                          /*history_depth=*/3);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, GeneralizedAlpha_FirstOrder_RhoInfSweep_IsSecondOrderOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    // Use a smaller dt pair so the strongly dissipative ρ∞→0 endpoint is in the
    // asymptotic convergence regime (avoid accidental cancellation at coarser dt).
    const double dt1 = 0.05;
    const double dt2 = 0.025;

    const std::vector<double> rho_values = {0.0, 0.2, 0.5, 0.9, 1.0};

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    for (double rho_inf : rho_values) {
        const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, lambda,
                                              /*history_depth=*/3,
                                              /*controller=*/{},
                                              /*generalized_alpha_rho_inf=*/rho_inf,
                                              /*dg_degree=*/1,
                                              /*cg_degree=*/2,
                                              svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                              /*collocation_max_outer_iterations=*/4,
                                              /*collocation_outer_tolerance=*/0.0,
                                              /*exact_initial_history=*/true);
        const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, lambda,
                                              /*history_depth=*/3,
                                              /*controller=*/{},
                                              /*generalized_alpha_rho_inf=*/rho_inf,
                                              /*dg_degree=*/1,
                                              /*cg_degree=*/2,
                                              svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                              /*collocation_max_outer_iterations=*/4,
                                              /*collocation_outer_tolerance=*/0.0,
                                              /*exact_initial_history=*/true);
        ASSERT_EQ(u_dt1.size(), 4u);
        ASSERT_EQ(u_dt2.size(), 4u);

        const double e1 = relativeL2Error(u_dt1, exact);
        const double e2 = relativeL2Error(u_dt2, exact);
        const double p = std::log(e1 / e2) / std::log(2.0);
        EXPECT_GT(p, 1.6) << "rho_inf=" << rho_inf << " e1=" << e1 << " e2=" << e2;
    }
}

TEST(TimeLoopEquivalences, DG0_MatchesBackwardEulerOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.4;
    const double dt = 0.1;

    const auto u_be = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt, t_end, lambda);
    const auto u_dg0 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG0, dt, t_end, lambda);
    ASSERT_EQ(u_be.size(), u_dg0.size());

    for (std::size_t i = 0; i < u_be.size(); ++i) {
        EXPECT_NEAR(u_be[i], u_dg0[i], 1e-12);
    }
}

TEST(TimeLoopEquivalences, CG1_MatchesThetaHalfOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.4;
    const double dt = 0.1;

    const auto u_theta = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt, t_end, lambda);
    const auto u_cg1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG1, dt, t_end, lambda);
    ASSERT_EQ(u_theta.size(), u_cg1.size());

		    for (std::size_t i = 0; i < u_theta.size(); ++i) {
		        EXPECT_NEAR(u_theta[i], u_cg1[i], 1e-12);
		    }
		}

TEST(TimeLoopEquivalences, ThetaMethod_Theta1_MatchesBackwardEulerOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.4;
    const double dt = 0.1;

    const auto u_be = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt, t_end, lambda);
    const auto u_theta1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod,
                                             dt,
                                             t_end,
                                             lambda,
                                             /*history_depth=*/2,
                                             /*controller=*/{},
                                             /*generalized_alpha_rho_inf=*/1.0,
                                             /*dg_degree=*/1,
                                             /*cg_degree=*/2,
                                             svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                             /*collocation_max_outer_iterations=*/4,
                                             /*collocation_outer_tolerance=*/0.0,
                                             /*exact_initial_history=*/false,
                                             /*theta=*/1.0);
    ASSERT_EQ(u_be.size(), u_theta1.size());

    for (std::size_t i = 0; i < u_be.size(); ++i) {
        EXPECT_NEAR(u_be[i], u_theta1[i], 1e-12);
    }
}

		TEST(TimeLoopEquivalences, DG1_StageGaussSeidelIsCloseToMonolithicOnReaction)
		{
		#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
		    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
	#endif
	    const double lambda = 1.0;
	    const double t_end = 0.4;
	    const double dt = 0.1;

	    const auto u_monolithic = runReactionProblem(
	        svmp::FE::timestepping::SchemeKind::DG1,
	        dt,
	        t_end,
	        lambda,
	        /*history_depth=*/2,
	        /*controller=*/{},
	        /*generalized_alpha_rho_inf=*/1.0,
	        /*dg_degree=*/1,
	        /*cg_degree=*/2,
	        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic);
	    const auto u_gs = runReactionProblem(
	        svmp::FE::timestepping::SchemeKind::DG1,
	        dt,
	        t_end,
	        lambda,
	        /*history_depth=*/2,
	        /*controller=*/{},
	        /*generalized_alpha_rho_inf=*/1.0,
	        /*dg_degree=*/1,
	        /*cg_degree=*/2,
	        svmp::FE::timestepping::CollocationSolveStrategy::StageGaussSeidel,
	        /*collocation_max_outer_iterations=*/20,
	        /*collocation_outer_tolerance=*/0.0);
	    ASSERT_EQ(u_monolithic.size(), u_gs.size());

	    const double e = relativeL2Error(u_gs, u_monolithic);
	    EXPECT_LT(e, 1e-5);
	}

	TEST(TimeLoopVSVO_BDF, AdaptsDtOnReaction)
	{
	#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
	    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 5.0;
    const double t_end = 1.0;
    const double dt0 = 0.5;

    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    ctrl_opts.abs_tol = 1e-10;
    ctrl_opts.rel_tol = 1e-6;
    ctrl_opts.min_order = 1;
    ctrl_opts.max_order = 3;
    ctrl_opts.initial_order = 1;
    ctrl_opts.max_retries = 8;
    ctrl_opts.safety = 0.9;
    ctrl_opts.min_factor = 0.2;
    ctrl_opts.max_factor = 2.0;
    ctrl_opts.increase_order_threshold = 0.05;

    auto controller = std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
    ASSERT_NE(controller.get(), nullptr);

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

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/5);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), u0);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt0);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
    opts.step_controller = controller;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    std::vector<std::pair<double, double>> dt_updates;
    callbacks.on_dt_updated = [&dt_updates](double oldv, double newv, int, int) {
        dt_updates.emplace_back(oldv, newv);
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, callbacks);
    if (!rep.success) {
        std::cerr << "TimeLoopVSVO_BDF dt2 oscillator failure message: " << rep.message << std::endl;
    }
    EXPECT_TRUE(rep.success) << rep.message;
    EXPECT_NEAR(rep.final_time, t_end, 1e-12) << rep.message;
    EXPECT_FALSE(dt_updates.empty());

    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }
	    const auto approx = getVectorByDof(history.uPrev());
	    const double err = relativeL2Error(approx, exact);
	    if (err >= 5e-4) {
	        double dt_min = dt0;
	        double dt_max = dt0;
	        for (const auto& p : dt_updates) {
	            dt_min = std::min(dt_min, std::min(p.first, p.second));
	            dt_max = std::max(dt_max, std::max(p.first, p.second));
	        }
	        std::cerr << "TimeLoopVSVO_BDF.AdaptsDtOnReaction diagnostics: "
	                  << "steps_taken=" << rep.steps_taken
	                  << " final_time=" << rep.final_time
	                  << " dt_prev=" << history.dtPrev()
	                  << " dt_updates=" << dt_updates.size()
	                  << " dt_min=" << dt_min
	                  << " dt_max=" << dt_max
	                  << std::endl;
	    }
	    EXPECT_LT(err, 5e-4);
	}

TEST(TimeLoopVSVO_BDF, RestartRequiresValidDtHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.2;
    const double dt0 = 0.1;

    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    ctrl_opts.abs_tol = 1e-6;
    ctrl_opts.rel_tol = 1e-4;
    ctrl_opts.min_order = 1;
    ctrl_opts.max_order = 2;
    ctrl_opts.initial_order = 1;
    ctrl_opts.max_retries = 2;
    auto controller = std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
    ASSERT_NE(controller.get(), nullptr);

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

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/3);
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), std::vector<Real>(static_cast<std::size_t>(n_dofs), 0.0));
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt0);

    // Simulate a restart with existing history but missing dtHistory() entries.
    history.setStepIndex(2);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
    opts.step_controller = controller;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 6;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    EXPECT_THROW((void)loop.run(transient, *factory, *linear, history), svmp::FE::InvalidArgumentException);
}

TEST(TimeLoopVSVO_BDF, RestartMatchesContinuousRunOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const double t_split = 0.5;
    const double dt = 0.05;
    const int order = 4;

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

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};

    auto makeHistory = [&](double dt0) {
        auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/order + 1);
        for (int k = 1; k <= history.historyDepth(); ++k) {
            const double t_k = -static_cast<double>(k - 1) * dt0;
            const double scale = std::exp(-lambda * t_k);
            std::vector<Real> u_k(u0.size(), 0.0);
            for (std::size_t i = 0; i < u0.size(); ++i) {
                u_k[i] = static_cast<Real>(static_cast<double>(u0[i]) * scale);
            }
            setVectorByDof(history.uPrevK(k), u_k);
        }
        history.resetCurrentToPrevious();
        history.setPrevDt(dt0);
        return history;
    };

    auto run = [&](double t0, double t1, svmp::FE::timestepping::TimeHistory& history) {
        svmp::FE::timestepping::TimeLoopOptions opts;
        opts.t0 = t0;
        opts.t_end = t1;
        opts.dt = dt;
        opts.max_steps = 2000;
        opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
        opts.step_controller = makeFixedVsvoBdfController(order, dt);
        opts.newton.residual_op = "op";
        opts.newton.jacobian_op = "op";
        opts.newton.max_iterations = 8;
        opts.newton.abs_tolerance = 1e-12;
        opts.newton.rel_tolerance = 0.0;

        svmp::FE::timestepping::TimeLoop loop(opts);
        const auto rep = loop.run(transient, *factory, *linear, history);
        EXPECT_TRUE(rep.success) << rep.message;
        EXPECT_NEAR(rep.final_time, t1, 1e-12) << rep.message;
    };

    // Continuous run.
    auto history_full = makeHistory(dt);
    run(/*t0=*/0.0, /*t1=*/t_end, history_full);
    const auto x_full = getVectorByDof(history_full.uPrev());

    // Restarted run: split at t_split and copy state into a fresh TimeHistory.
    auto history_part1 = makeHistory(dt);
    run(/*t0=*/0.0, /*t1=*/t_split, history_part1);

    auto history_restart = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/order + 1);
    for (int k = 1; k <= history_restart.historyDepth(); ++k) {
        auto dst = history_restart.uPrevK(k).localSpan();
        auto src = history_part1.uPrevK(k).localSpan();
        ASSERT_EQ(dst.size(), src.size());
        std::copy(src.begin(), src.end(), dst.begin());
    }
    history_restart.resetCurrentToPrevious();
    history_restart.setStepIndex(history_part1.stepIndex());
    history_restart.setTime(history_part1.time());
    history_restart.setPrevDt(history_part1.dtPrev());
    history_restart.setDtHistory(std::vector<double>(history_part1.dtHistory().begin(), history_part1.dtHistory().end()));

    run(/*t0=*/t_split, /*t1=*/t_end, history_restart);
    const auto x_restart = getVectorByDof(history_restart.uPrev());

    ASSERT_EQ(x_full.size(), x_restart.size());
    for (std::size_t i = 0; i < x_full.size(); ++i) {
        EXPECT_NEAR(x_full[i], x_restart[i], 1e-12);
    }

    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double err_full = relativeL2Error(x_full, exact);
    const double err_restart = relativeL2Error(x_restart, exact);
    EXPECT_LT(err_full, 1e-5);
    EXPECT_LT(err_restart, 1e-5);
}

TEST(TimeLoopVSVO_BDF, AdaptsDtOnDt2Oscillator)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 2.0;
    const double t_end = 1.0;
    const double dt0 = 0.25;

    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    ctrl_opts.abs_tol = 1e-4;
    ctrl_opts.rel_tol = 1e-3;
    ctrl_opts.min_order = 1;
    ctrl_opts.max_order = 3;
    ctrl_opts.initial_order = 1;
    ctrl_opts.max_retries = 12;
    ctrl_opts.safety = 0.9;
    ctrl_opts.min_factor = 0.2;
    ctrl_opts.max_factor = 2.0;
    ctrl_opts.increase_order_threshold = 0.05;

    auto controller = std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
    ASSERT_NE(controller.get(), nullptr);

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
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 n_dofs,
                                                                 /*history_depth=*/5,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    ASSERT_EQ(u0.size(), static_cast<std::size_t>(n_dofs));

    // Exact solution with v0 = omega*u0 => u(t)=u0*(cos(omega t)+sin(omega t)).
    std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        v0[i] = static_cast<Real>(omega * static_cast<double>(u0[i]));
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    for (int k = 1; k <= history.historyDepth(); ++k) {
        const double t = -static_cast<double>(k - 1) * dt0;
        const double c = std::cos(omega * t);
        const double s = std::sin(omega * t);
        std::vector<Real> uk(u0.size(), 0.0);
        for (std::size_t i = 0; i < u0.size(); ++i) {
            uk[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c + s));
        }
        setVectorByDof(history.uPrevK(k), uk);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt0);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
    opts.step_controller = controller;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 10;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    std::vector<std::pair<double, double>> dt_updates;
    callbacks.on_dt_updated = [&dt_updates](double oldv, double newv, int, int) {
        dt_updates.emplace_back(oldv, newv);
    };

	    svmp::FE::timestepping::TimeLoop loop(opts);
	    const auto rep = loop.run(transient, *factory, *linear, history, callbacks);
	    if (!rep.success) {
	        double dt_min = dt0;
	        double dt_max = dt0;
	        for (const auto& p : dt_updates) {
	            dt_min = std::min(dt_min, std::min(p.first, p.second));
	            dt_max = std::max(dt_max, std::max(p.first, p.second));
	        }
	        std::cerr << "TimeLoopVSVO_BDF.AdaptsDtOnDt2Oscillator diagnostics: "
	                  << "steps_taken=" << rep.steps_taken
	                  << " final_time=" << rep.final_time
	                  << " dt_prev=" << history.dtPrev()
	                  << " dt_updates=" << dt_updates.size()
	                  << " dt_min=" << dt_min
	                  << " dt_max=" << dt_max
	                  << std::endl;
	    }
	    EXPECT_TRUE(rep.success) << rep.message;
	    EXPECT_NEAR(rep.final_time, t_end, 1e-12) << rep.message;
	    EXPECT_FALSE(dt_updates.empty());

    const double c_end = std::cos(omega * t_end);
    const double s_end = std::sin(omega * t_end);
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c_end + s_end));
    }
    const auto approx = getVectorByDof(history.uPrev());
    const double err = relativeL2Error(approx, exact);
    EXPECT_LT(err, 0.1);
}
