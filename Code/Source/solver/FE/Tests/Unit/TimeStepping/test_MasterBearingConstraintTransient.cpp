/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_MasterBearingConstraintTransient.cpp
 * @brief Minimal fixture: one master-bearing (MPC) constraint through the
 *        full transient Newton pipeline.
 *
 * Reproduction harness for the small-cut-aggregation blocker: a 4-DOF
 * single-tet reaction problem du/dt + lambda*u = f with one affine line
 * u_3 = 0.5*u_0 + 0.5*u_1. The condensed problem is linear, so every scheme
 * must converge in O(1) Newton iterations with a full step; any line-search
 * rejection or residual growth along the Newton direction indicates an
 * inconsistency between the condensed Jacobian/residual and the
 * trial-state residual evaluation for master-bearing constraints
 * (plain Dirichlet lines cannot see it because their increment is zero).
 *
 * SCOPE: this exercises the core CELL-term pipeline only (constraints fixed
 * at setup, single scalar field, interpreter kernels, square element
 * blocks). It does NOT cover interior-face or generated-interface
 * insertions, the fused combined-insert path, JIT kernels, multi-field
 * offsets, post-setup constraint-structure changes (sparsity augmentation
 * runs at setup only), or out-of-pattern matrix writes — see
 * Documentation/plan_ghost_penalty_eigen_calibration_20260611.md for the
 * uncovered failure modes found on the d18 aggregation runs.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"

#include "Constraints/AffineConstraints.h"
#include "Constraints/SystemConstraint.h"

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

#include <cmath>
#include <memory>
#include <vector>

namespace {

using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::ElementType;
namespace ts_test = svmp::FE::timestepping::test;

/// u_slave = 0.5*u_m0 + 0.5*u_m1 + inhomogeneity.
class AverageMpcConstraint final : public svmp::FE::constraints::ISystemConstraint {
public:
    AverageMpcConstraint(GlobalIndex slave, GlobalIndex m0, GlobalIndex m1, Real inhomogeneity)
        : slave_(slave), m0_(m0), m1_(m1), inhomogeneity_(inhomogeneity)
    {
    }

    void apply(const svmp::FE::systems::FESystem& /*system*/,
               svmp::FE::constraints::AffineConstraints& constraints) override
    {
        constraints.addLine(slave_);
        constraints.addEntry(slave_, m0_, 0.5);
        constraints.addEntry(slave_, m1_, 0.5);
        if (inhomogeneity_ != Real(0)) {
            constraints.setInhomogeneity(slave_, inhomogeneity_);
        }
    }

    bool updateValues(const svmp::FE::systems::FESystem& /*system*/,
                      svmp::FE::constraints::AffineConstraints& /*constraints*/,
                      double /*time*/,
                      double /*dt*/) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

    [[nodiscard]] svmp::FE::systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        return {};
    }

private:
    GlobalIndex slave_;
    GlobalIndex m0_;
    GlobalIndex m1_;
    Real inhomogeneity_;
};

struct MpcRunResult {
    bool ran{false};
    bool success{false};
    int max_newton_iterations{0};
    double max_solution_error{0.0};
    double max_constraint_violation{0.0};
};

/// du/dt + lambda*u = f on a single tet with u_3 = avg(u_0, u_1).
/// Exact per-DOF solution: u_i(t) = f/lambda + (u0_i - f/lambda) exp(-lambda t),
/// which satisfies the constraint for all t when u0_3 = avg(u0_0, u0_1).
[[maybe_unused]] MpcRunResult runMpcReactionProblem(
                                   svmp::FE::timestepping::SchemeKind scheme,
                                   double generalized_alpha_rho_inf,
                                   double dt,
                                   double t_end,
                                   double lambda,
                                   double source)
{
    MpcRunResult result;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v +
                       (u * v) * static_cast<Real>(lambda) -
                       v * static_cast<Real>(source))
                          .dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
        std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    sys.addSystemConstraint(
        std::make_unique<AverageMpcConstraint>(3, 0, 1, Real(0)));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);
    if (!sys.isSetup()) {
        ADD_FAILURE() << "FESystem::setup did not mark the system as setup";
        return result;
    }

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available";
        return result;
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return result;
    }

    const auto n_dofs = sys.dofHandler().getNumDofs();
    if (n_dofs != 4) {
        ADD_FAILURE() << "expected 4 DOFs, got " << n_dofs;
        return result;
    }
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, 2);

    // u0_3 = avg(u0_0, u0_1) so the constrained initial state is consistent.
    const std::vector<Real> u0 = {1.0, -0.5, 0.75, 0.25};
    for (int k = 1; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), u0);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    opts.newton.use_line_search = true;
    opts.newton.line_search_fail_on_no_reduction = true;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    int max_iters = 0;
    callbacks.on_nonlinear_done =
        [&max_iters](const svmp::FE::timestepping::TimeHistory&,
                     const svmp::FE::timestepping::NewtonReport& nr) {
            max_iters = std::max(max_iters, nr.iterations);
        };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << "TimeLoop threw: " << e.what();
        return result;
    }
    result.ran = true;
    result.success = rep.success;
    result.max_newton_iterations = max_iters;

    // Compare against the exact solution at the final time.
    std::vector<Real> u_num(static_cast<std::size_t>(n_dofs), Real(0));
    {
        const auto span = history.uSpan();
        for (std::size_t i = 0; i < u_num.size(); ++i) {
            u_num[i] = span[i];
        }
    }
    const double t = history.time();
    const double u_inf = lambda > 0.0 ? source / lambda : 0.0;
    double max_err = 0.0;
    for (std::size_t i = 0; i < u_num.size(); ++i) {
        const double exact =
            u_inf + (static_cast<double>(u0[i]) - u_inf) * std::exp(-lambda * t);
        max_err = std::max(max_err, std::abs(static_cast<double>(u_num[i]) - exact));
    }
    result.max_solution_error = max_err;
    result.max_constraint_violation = std::abs(
        static_cast<double>(u_num[3]) -
        0.5 * (static_cast<double>(u_num[0]) + static_cast<double>(u_num[1])));
    return result;
}

TEST(MasterBearingConstraintTransient, BackwardEulerMpcConvergesAndTracksExact)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "This transient solve requires the Eigen backend.";
#else
    const auto result = runMpcReactionProblem(
        svmp::FE::timestepping::SchemeKind::BackwardEuler,
        /*rho_inf=*/1.0,
        /*dt=*/0.05,
        /*t_end=*/0.5,
        /*lambda=*/2.0,
        /*source=*/0.0);
    ASSERT_TRUE(result.ran);
    EXPECT_TRUE(result.success);
    EXPECT_LE(result.max_newton_iterations, 3)
        << "linear condensed problem should not need line-search rescue";
    EXPECT_LT(result.max_solution_error, 5e-2);
    EXPECT_LT(result.max_constraint_violation, 1e-10);
#endif
}

TEST(MasterBearingConstraintTransient, GeneralizedAlphaMpcConvergesAndTracksExact)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "This transient solve requires the Eigen backend.";
#else
    const auto result = runMpcReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*rho_inf=*/0.5,
        /*dt=*/0.05,
        /*t_end=*/0.5,
        /*lambda=*/2.0,
        /*source=*/0.0);
    ASSERT_TRUE(result.ran);
    EXPECT_TRUE(result.success);
    EXPECT_LE(result.max_newton_iterations, 3);
    EXPECT_LT(result.max_solution_error, 2e-2);
    EXPECT_LT(result.max_constraint_violation, 1e-10);
#endif
}

TEST(MasterBearingConstraintTransient, GeneralizedAlphaMpcWithConstantSource)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "This transient solve requires the Eigen backend.";
#else
    // Constant source mimics the body-force/gravity path of the dam-break
    // configuration where the aggregation blocker was observed.
    const auto result = runMpcReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*rho_inf=*/0.5,
        /*dt=*/0.05,
        /*t_end=*/0.5,
        /*lambda=*/2.0,
        /*source=*/3.0);
    ASSERT_TRUE(result.ran);
    EXPECT_TRUE(result.success);
    EXPECT_LE(result.max_newton_iterations, 3);
    EXPECT_LT(result.max_solution_error, 2e-2);
    EXPECT_LT(result.max_constraint_violation, 1e-10);
#endif
}

} // namespace
