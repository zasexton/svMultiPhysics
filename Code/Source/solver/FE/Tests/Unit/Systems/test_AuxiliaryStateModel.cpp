/**
 * @file test_AuxiliaryStateModel.cpp
 * @brief Unit tests for AuxiliaryStateModel, AuxiliaryDerivativeProvider,
 *        and AuxiliaryStateStepper.
 */

#include <gtest/gtest.h>

#include "Systems/AuxiliaryStateModel.h"
#include "Systems/AuxiliaryDerivativeProvider.h"
#include "Systems/AuxiliaryStateStepper.h"
#include "Spaces/H1Space.h"

#include <cmath>
#include <memory>
#include <vector>

using svmp::FE::Real;
using namespace svmp::FE::systems;

// ============================================================================
//  Test model: scalar linear decay  dx/dt = -k*x
//  Residual form: F(xdot, x) = xdot + k*x = 0
// ============================================================================

class ScalarDecayModel : public AuxiliaryStateModel {
public:
    explicit ScalarDecayModel(Real k = 1.0) : k_(k) {}

    std::string modelName() const override { return "ScalarDecay"; }
    int dimension() const override { return 1; }

    AuxiliaryStructuralMetadata structuralMetadata() const override
    {
        AuxiliaryStructuralMetadata meta;
        meta.variable_kinds = {AuxiliaryVariableKind::Differential};
        return meta;
    }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override
    {
        // F = xdot + k*x
        req.residual[0] = ctx.xdot[0] + k_ * ctx.x[0];
    }

    bool hasAnalyticJacobian() const override { return true; }

    void evaluateJacobian(const AuxiliaryLocalContext& /*ctx*/,
                          AuxiliaryJacobianRequest& req) const override
    {
        // dF/dx = k
        if (!req.dF_dx.empty()) req.dF_dx[0] = k_;
        // dF/dxdot = 1
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 1.0;
    }

private:
    Real k_;
};

// ============================================================================
//  Test model: 2D coupled ODE  dx/dt = -a*x + b*y,  dy/dt = c*x - d*y
// ============================================================================

class CoupledODE2D : public AuxiliaryStateModel {
public:
    CoupledODE2D(Real a, Real b, Real c, Real d) : a_(a), b_(b), c_(c), d_(d) {}

    std::string modelName() const override { return "CoupledODE2D"; }
    int dimension() const override { return 2; }

    AuxiliaryStructuralMetadata structuralMetadata() const override
    {
        AuxiliaryStructuralMetadata meta;
        meta.variable_kinds = {AuxiliaryVariableKind::Differential,
                               AuxiliaryVariableKind::Differential};
        return meta;
    }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override
    {
        // F0 = xdot[0] + a*x[0] - b*x[1]
        // F1 = xdot[1] - c*x[0] + d*x[1]
        req.residual[0] = ctx.xdot[0] + a_ * ctx.x[0] - b_ * ctx.x[1];
        req.residual[1] = ctx.xdot[1] - c_ * ctx.x[0] + d_ * ctx.x[1];
    }

    bool hasAnalyticJacobian() const override { return true; }

    void evaluateJacobian(const AuxiliaryLocalContext& /*ctx*/,
                          AuxiliaryJacobianRequest& req) const override
    {
        if (!req.dF_dx.empty()) {
            req.dF_dx[0] = a_;   req.dF_dx[1] = -b_;
            req.dF_dx[2] = -c_;  req.dF_dx[3] = d_;
        }
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) {
            req.dF_dxdot[0] = 1.0; req.dF_dxdot[1] = 0.0;
            req.dF_dxdot[2] = 0.0; req.dF_dxdot[3] = 1.0;
        }
    }

private:
    Real a_, b_, c_, d_;
};

// ============================================================================
//  Test model: mixed DAE (differential + algebraic)
//  x' = -x + z,  0 = x + z - 1  (algebraic: z = 1 - x)
// ============================================================================

class MixedDAEModel : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "MixedDAE"; }
    int dimension() const override { return 2; }

    AuxiliaryStructuralMetadata structuralMetadata() const override
    {
        AuxiliaryStructuralMetadata meta;
        meta.variable_kinds = {AuxiliaryVariableKind::Differential,
                               AuxiliaryVariableKind::Algebraic};
        meta.dae_index_hint = 1;
        return meta;
    }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override
    {
        const Real x = ctx.x[0];
        const Real z = ctx.x[1];
        // F0 = xdot[0] + x - z  (differential)
        // F1 = x + z - 1        (algebraic)
        req.residual[0] = ctx.xdot[0] + x - z;
        req.residual[1] = x + z - 1.0;
    }

    bool hasAnalyticJacobian() const override { return true; }

    void evaluateJacobian(const AuxiliaryLocalContext& /*ctx*/,
                          AuxiliaryJacobianRequest& req) const override
    {
        if (!req.dF_dx.empty()) {
            req.dF_dx[0] = 1.0;  req.dF_dx[1] = -1.0;
            req.dF_dx[2] = 1.0;  req.dF_dx[3] = 1.0;
        }
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) {
            req.dF_dxdot[0] = 1.0; req.dF_dxdot[1] = 0.0;
            req.dF_dxdot[2] = 0.0; req.dF_dxdot[3] = 0.0;
        }
    }
};

// ============================================================================
//  Test model: purely algebraic (no differential rows)
//  g(x) = x^2 - 4 = 0  =>  x = 2
// ============================================================================

class PureAlgebraicModel : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "PureAlgebraic"; }
    int dimension() const override { return 1; }

    AuxiliaryStructuralMetadata structuralMetadata() const override
    {
        AuxiliaryStructuralMetadata meta;
        meta.variable_kinds = {AuxiliaryVariableKind::Algebraic};
        return meta;
    }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override
    {
        req.residual[0] = ctx.x[0] * ctx.x[0] - 4.0;
    }

    bool hasAnalyticJacobian() const override { return true; }

    void evaluateJacobian(const AuxiliaryLocalContext& ctx,
                          AuxiliaryJacobianRequest& req) const override
    {
        if (!req.dF_dx.empty()) req.dF_dx[0] = 2.0 * ctx.x[0];
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 0.0;
    }
};

// ============================================================================
//  Model interface tests
// ============================================================================

TEST(AuxiliaryStateModel, ScalarDecayMetadata)
{
    ScalarDecayModel model(1.0);
    EXPECT_EQ(model.modelName(), "ScalarDecay");
    EXPECT_EQ(model.dimension(), 1);

    auto meta = model.structuralMetadata();
    ASSERT_EQ(meta.variable_kinds.size(), 1u);
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);
}

TEST(AuxiliaryStateModel, MixedDAEMetadata)
{
    MixedDAEModel model;
    EXPECT_EQ(model.dimension(), 2);

    auto meta = model.structuralMetadata();
    ASSERT_EQ(meta.variable_kinds.size(), 2u);
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[1], AuxiliaryVariableKind::Algebraic);
    EXPECT_EQ(meta.dae_index_hint, 1);
}

TEST(AuxiliaryStateModel, PureAlgebraicMetadata)
{
    PureAlgebraicModel model;
    auto meta = model.structuralMetadata();
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Algebraic);
}

TEST(AuxiliaryStateModel, ResidualEvaluation)
{
    ScalarDecayModel model(2.0); // dx/dt = -2x

    std::vector<Real> x = {3.0};
    std::vector<Real> xdot = {0.0};
    std::vector<Real> residual(1);

    AuxiliaryLocalContext ctx;
    ctx.x = x;
    ctx.xdot = xdot;

    AuxiliaryResidualRequest req;
    req.residual = residual;

    model.evaluateResidual(ctx, req);
    // F = 0 + 2*3 = 6
    EXPECT_DOUBLE_EQ(residual[0], 6.0);
}

TEST(AuxiliaryStateModel, JacobianEvaluation)
{
    ScalarDecayModel model(2.0);

    std::vector<Real> x = {3.0};
    std::vector<Real> xdot = {0.0};

    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot;

    std::vector<Real> dFdx(1), dFdxdot(1);
    AuxiliaryJacobianRequest jreq;
    jreq.dF_dx = dFdx;
    jreq.dF_dxdot = dFdxdot;
    jreq.want_dF_dxdot = true;
    jreq.n = 1;

    model.evaluateJacobian(ctx, jreq);
    EXPECT_DOUBLE_EQ(dFdx[0], 2.0);    // k
    EXPECT_DOUBLE_EQ(dFdxdot[0], 1.0); // identity
}

TEST(AuxiliaryStateModel, OptionalHooksDefaultToFalse)
{
    ScalarDecayModel model;
    EXPECT_TRUE(model.hasAnalyticJacobian());
    EXPECT_FALSE(model.hasAnalyticHessian());
    EXPECT_FALSE(model.hasConsistentInitialization());
    EXPECT_FALSE(model.hasEventFunctions());
    EXPECT_FALSE(model.hasNonsmoothHooks());
    EXPECT_FALSE(model.hasResidualExpressions());
    EXPECT_FALSE(model.hasMassMatrix());
}

// ============================================================================
//  Derivative provider tests
// ============================================================================

TEST(AuxiliaryDerivativeProvider, AnalyticOverride)
{
    ScalarDecayModel model(2.0);
    AuxiliaryDerivativePolicy policy;

    AuxiliaryDerivativeProvider provider;
    provider.setup(model, policy);

    EXPECT_TRUE(provider.isSetup());
    EXPECT_TRUE(provider.hasAnalyticJacobian());
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Analytic);
    EXPECT_EQ(provider.artifact().source, AuxiliaryDerivativeSource::Analytic);
}

TEST(AuxiliaryDerivativeProvider, FiniteDifferenceFallback)
{
    // Model without analytic Jacobian
    class NoJacModel : public AuxiliaryStateModel {
    public:
        std::string modelName() const override { return "NoJac"; }
        int dimension() const override { return 1; }
        AuxiliaryStructuralMetadata structuralMetadata() const override
        {
            AuxiliaryStructuralMetadata m;
            m.variable_kinds = {AuxiliaryVariableKind::Differential};
            return m;
        }
        void evaluateResidual(const AuxiliaryLocalContext& ctx,
                              AuxiliaryResidualRequest& req) const override
        {
            req.residual[0] = ctx.xdot[0] + 2.0 * ctx.x[0];
        }
    };

    NoJacModel model;
    AuxiliaryDerivativePolicy policy;
    policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;

    AuxiliaryDerivativeProvider provider;
    provider.setup(model, policy);

    EXPECT_TRUE(provider.isSetup());
    EXPECT_FALSE(provider.hasAnalyticJacobian());

    // Evaluate FD Jacobian
    std::vector<Real> x = {3.0};
    std::vector<Real> xdot = {0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot;

    std::vector<Real> dFdx(1), dFdxdot(1);
    AuxiliaryJacobianRequest jreq;
    jreq.dF_dx = dFdx;
    jreq.dF_dxdot = dFdxdot;
    jreq.want_dF_dxdot = true;
    jreq.n = 1;

    provider.evaluateJacobian(model, ctx, jreq);

    EXPECT_NEAR(dFdx[0], 2.0, 1e-5);    // dF/dx = k = 2
    EXPECT_NEAR(dFdxdot[0], 1.0, 1e-5); // dF/dxdot = 1
}

// ============================================================================
//  Stepper factory tests
// ============================================================================

TEST(AuxiliaryStateStepper, FactoryCreatesKnownMethods)
{
    auto fe = createStepper("ForwardEuler");
    EXPECT_EQ(fe->methodName(), "ForwardEuler");
    EXPECT_FALSE(fe->requiresJacobian());

    auto be = createStepper("BackwardEuler");
    EXPECT_EQ(be->methodName(), "BackwardEuler");
    EXPECT_TRUE(be->requiresJacobian());

    auto rk4 = createStepper("RK4");
    EXPECT_EQ(rk4->methodName(), "RK4");

    auto bdf2 = createStepper("BDF2");
    EXPECT_EQ(bdf2->methodName(), "BDF2");
    EXPECT_EQ(bdf2->requiredHistoryDepth(), 1);
}

TEST(AuxiliaryStateStepper, FactoryThrowsOnUnknown)
{
    EXPECT_THROW(createStepper("Bogus"), svmp::FE::InvalidArgumentException);
}

// ============================================================================
//  Stepper integration tests: scalar decay  dx/dt = -x
// ============================================================================

TEST(AuxiliaryStateStepper, ForwardEuler_ScalarDecay)
{
    ScalarDecayModel model(1.0);
    AuxiliaryDerivativeProvider deriv;
    deriv.setup(model, {});

    auto stepper = createStepper("ForwardEuler");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    std::vector<Real> x = {1.0};
    const std::vector<Real> x0 = {1.0};
    const Real dt = 0.1;

    auto result = stepper->advance(model, deriv, x, x0, {}, {}, {}, 0.0, dt);

    EXPECT_TRUE(result.converged);
    // Forward Euler: x = 1 + 0.1*(-1) = 0.9
    EXPECT_NEAR(x[0], 0.9, 1e-12);
}

TEST(AuxiliaryStateStepper, BackwardEuler_ScalarDecay)
{
    ScalarDecayModel model(1.0);
    AuxiliaryDerivativeProvider deriv;
    deriv.setup(model, {});

    auto stepper = createStepper("BackwardEuler");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    std::vector<Real> x = {1.0};
    const std::vector<Real> x0 = {1.0};
    const Real dt = 0.1;

    auto result = stepper->advance(model, deriv, x, x0, {}, {}, {}, 0.0, dt);

    EXPECT_TRUE(result.converged);
    // Backward Euler: x = 1/(1+dt) = 1/1.1 ≈ 0.909091
    EXPECT_NEAR(x[0], 1.0 / 1.1, 1e-10);
}

TEST(AuxiliaryStateStepper, RK4_ScalarDecay)
{
    ScalarDecayModel model(1.0);
    AuxiliaryDerivativeProvider deriv;
    deriv.setup(model, {});

    auto stepper = createStepper("RK4");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    std::vector<Real> x = {1.0};
    const std::vector<Real> x0 = {1.0};
    const Real dt = 0.1;

    auto result = stepper->advance(model, deriv, x, x0, {}, {}, {}, 0.0, dt);

    EXPECT_TRUE(result.converged);
    // RK4 should be very close to exact exp(-0.1) ≈ 0.904837
    EXPECT_NEAR(x[0], std::exp(-0.1), 1e-6);
}

// ============================================================================
//  Multi-component stepper test: 2D coupled ODE
// ============================================================================

TEST(AuxiliaryStateStepper, BackwardEuler_CoupledODE2D)
{
    // dx/dt = -x + y,  dy/dt = x - y
    CoupledODE2D model(1.0, 1.0, 1.0, 1.0);
    AuxiliaryDerivativeProvider deriv;
    deriv.setup(model, {});

    auto stepper = createStepper("BackwardEuler");
    AuxiliaryStepperSpec spec;
    stepper->setup(2, spec);

    std::vector<Real> x = {1.0, 0.0};
    const std::vector<Real> x0 = {1.0, 0.0};

    auto result = stepper->advance(model, deriv, x, x0, {}, {}, {}, 0.0, 0.1);

    EXPECT_TRUE(result.converged);
    // Both components should be between 0 and 1 (coupled relaxation)
    EXPECT_GT(x[0], 0.0);
    EXPECT_LT(x[0], 1.0);
    EXPECT_GT(x[1], 0.0);
    EXPECT_LT(x[1], 1.0);
    // Conservation: x + y should be close to 1.0 (sum preserved)
    EXPECT_NEAR(x[0] + x[1], 1.0, 1e-10);
}

// ============================================================================
//  Substepping test
// ============================================================================

TEST(AuxiliaryStateStepper, ForwardEuler_Substepping)
{
    ScalarDecayModel model(1.0);
    AuxiliaryDerivativeProvider deriv;
    deriv.setup(model, {});

    auto stepper = createStepper("ForwardEuler");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    std::vector<Real> x = {1.0};
    const std::vector<Real> x0 = {1.0};
    const Real dt = 0.1;

    // 10 substeps of dt/10 = 0.01 each
    auto result = stepper->advance(model, deriv, x, x0, {}, {}, {}, 0.0, dt, 10);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.substeps_taken, 10);
    // With substepping, result should be closer to exact
    EXPECT_NEAR(x[0], std::pow(0.99, 10), 1e-12);
}

// ============================================================================
//  BDF2 with history fallback
// ============================================================================

TEST(AuxiliaryStateStepper, BDF2_FallsBackToBackwardEulerWithoutHistory)
{
    ScalarDecayModel model(1.0);
    AuxiliaryDerivativeProvider deriv;
    deriv.setup(model, {});

    auto stepper = createStepper("BDF2");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    std::vector<Real> x = {1.0};
    const std::vector<Real> x0 = {1.0};

    // No history → falls back to BackwardEuler
    auto result = stepper->advance(model, deriv, x, x0, {}, {}, {}, 0.0, 0.1);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 1.0 / 1.1, 1e-10);
}

TEST(AuxiliaryStateStepper, BDF2_WithHistory)
{
    ScalarDecayModel model(1.0);
    AuxiliaryDerivativeProvider deriv;
    deriv.setup(model, {});

    auto stepper = createStepper("BDF2");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    // x_{n-1} = 1.0 (two steps back), x_n = 1/1.1 (one step back = committed)
    const Real x_nm1_val = 1.0;
    const Real x_n_val = 1.0 / 1.1;
    std::vector<Real> x = {x_n_val};
    const std::vector<Real> x_committed = {x_n_val};

    // History: one entry = x_{n-1}
    const std::vector<Real> hist0 = {x_nm1_val};
    const std::span<const Real> hist_span = hist0;
    const std::vector<std::span<const Real>> history = {hist_span};

    auto result = stepper->advance(model, deriv, x, x_committed,
                                   history, {}, {}, 0.1, 0.1);

    EXPECT_TRUE(result.converged);
    // BDF2 should give a different (more accurate) result than BE
    EXPECT_GT(x[0], 0.0);
    EXPECT_LT(x[0], x_n_val);
}

// ============================================================================
//  Per-block method selection via factory
// ============================================================================

TEST(AuxiliaryStateStepper, PerBlockMethodSelection)
{
    // Simulate two blocks with different methods
    auto stepper_a = createStepper("ForwardEuler");
    auto stepper_b = createStepper("BackwardEuler");

    EXPECT_EQ(stepper_a->methodName(), "ForwardEuler");
    EXPECT_EQ(stepper_b->methodName(), "BackwardEuler");
    EXPECT_FALSE(stepper_a->requiresJacobian());
    EXPECT_TRUE(stepper_b->requiresJacobian());
}

// ============================================================================
//  Symbolic Jacobian parity tests
// ============================================================================

#include "Systems/AuxiliaryModelBuilder.h"

TEST(AuxiliaryDerivativeProvider, SymbolicJacobian_ScalarDecay)
{
    using namespace svmp::FE::systems;

    // dx/dt = -k*x → dF/dx = -k (the RHS derivative)
    // The builder stores the RHS expr; symbolic diff gives d(RHS)/dx.
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .param("k")
        .ode("x", -modelParam("k") * modelState("x"))
        .build();

    // Symbolic provider (default for expression-defined models).
    AuxiliaryDerivativeProvider sym_provider;
    sym_provider.setup(*model, {});
    EXPECT_EQ(sym_provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);
    EXPECT_TRUE(sym_provider.hasSymbolicArtifacts());

    // FD provider for comparison.
    AuxiliaryDerivativePolicy fd_policy;
    fd_policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;
    AuxiliaryDerivativeProvider fd_provider;
    fd_provider.setup(*model, fd_policy);
    EXPECT_EQ(fd_provider.resolvedSource(), AuxiliaryDerivativeSource::FiniteDifference);

    // Evaluate at x=3.0, k=2.0
    std::vector<Real> x = {3.0};
    std::vector<Real> xdot = {0.0};
    std::vector<Real> params = {2.0}; // k=2
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.params = params; ctx.time = 0.0; ctx.dt = 0.1;

    std::vector<Real> sym_dFdx(1), fd_dFdx(1);
    AuxiliaryJacobianRequest sym_req, fd_req;
    sym_req.dF_dx = sym_dFdx; sym_req.n = 1;
    fd_req.dF_dx = fd_dFdx; fd_req.n = 1;

    sym_provider.evaluateJacobian(*model, ctx, sym_req);
    fd_provider.evaluateJacobian(*model, ctx, fd_req);

    // Residual F = xdot - rhs = xdot - (-k*x) = xdot + k*x.
    // dF/dx = k = 2 (both symbolic and FD).
    EXPECT_NEAR(sym_dFdx[0], 2.0, 1e-12);
    EXPECT_NEAR(fd_dFdx[0], 2.0, 1e-5);
}

TEST(AuxiliaryDerivativeProvider, SymbolicJacobian_2D_MixedDAE)
{
    using namespace svmp::FE::systems;

    // x' = -x + z,  0 = x + z - 1
    // Residual RHS stored by builder:
    //   row 0 (ODE): -x + z → dF0/dx = -1, dF0/dz = 1
    //   row 1 (alg): x + z - 1 → dF1/dx = 1, dF1/dz = 1
    auto model = AuxiliaryModelBuilder("dae")
        .state("x", AuxiliaryVariableKind::Differential)
        .state("z", AuxiliaryVariableKind::Algebraic)
        .ode("x", -modelState("x") + modelState("z"))
        .algebraic("z", modelState("x") + modelState("z") - svmp::FE::forms::FormExpr::constant(1.0))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> x = {0.5, 0.5};
    std::vector<Real> xdot = {0.0, 0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

    std::vector<Real> dFdx(4);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 2;

    provider.evaluateJacobian(*model, ctx, req);

    // Row 0 (ODE): F = xdot - (-x + z) = xdot + x - z
    // dF0/dx = 1, dF0/dz = -1
    EXPECT_NEAR(dFdx[0],  1.0, 1e-12); // dF0/dx
    EXPECT_NEAR(dFdx[1], -1.0, 1e-12); // dF0/dz
    // Row 1: d(x+z-1)/dx = 1, d(x+z-1)/dz = 1
    EXPECT_NEAR(dFdx[2],  1.0, 1e-12); // dF1/dx
    EXPECT_NEAR(dFdx[3],  1.0, 1e-12); // dF1/dz
}

TEST(AuxiliaryDerivativeProvider, SymbolicJacobian_NotFD_ForBuiltModels)
{
    using namespace svmp::FE::systems;

    auto model = AuxiliaryModelBuilder("test")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    // Default policy = Symbolic.
    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});

    // Verify it resolved to Symbolic, not FD.
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);
    EXPECT_TRUE(provider.hasSymbolicArtifacts());
    EXPECT_EQ(provider.artifact().source, AuxiliaryDerivativeSource::Symbolic);
}

TEST(AuxiliaryDerivativeProvider, SymbolicFallbackForUnsupportedOps)
{
    using namespace svmp::FE::systems;

    // Model with Gradient (FE differential op, not valid in aux residuals)
    // → symbolic diff throws → graceful FD fallback.
    auto model = AuxiliaryModelBuilder("nonlinear")
        .state("x")
        .ode("x", modelState("x").grad())
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {}); // Should NOT throw — graceful fallback.

    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::FiniteDifference);
    EXPECT_FALSE(provider.hasSymbolicArtifacts());
    EXPECT_EQ(provider.artifact().source, AuxiliaryDerivativeSource::FiniteDifference);
}

// ---------------------------------------------------------------------------
//  Rich symbolic differentiation tests
// ---------------------------------------------------------------------------

TEST(AuxiliaryDerivativeProvider, Symbolic_Power_ConstantExponent)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = x^3 → residual F = xdot - x^3
    // dF/dx = -3*x^2
    auto model = AuxiliaryModelBuilder("pow_model")
        .state("x")
        .ode("x", modelState("x").pow(fe::FormExpr::constant(3.0)))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> x = {2.0};
    std::vector<Real> xdot = {0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

    std::vector<Real> dFdx(1);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 1;
    provider.evaluateJacobian(*model, ctx, req);

    // dF/dx = d(xdot - x^3)/dx = -3*x^2 = -3*4 = -12
    EXPECT_NEAR(dFdx[0], -12.0, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Exp)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = exp(-x) → F = xdot - exp(-x)
    // dF/dx = exp(-x)
    auto model = AuxiliaryModelBuilder("exp_model")
        .state("x")
        .ode("x", (-modelState("x")).exp())
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> x = {1.0};
    std::vector<Real> xdot = {0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

    std::vector<Real> dFdx(1);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 1;
    provider.evaluateJacobian(*model, ctx, req);

    // d(xdot - exp(-x))/dx = -d(exp(-x))/dx = -(-exp(-x)) = exp(-x)
    EXPECT_NEAR(dFdx[0], std::exp(-1.0), 1e-10);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Log)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = log(x) → F = xdot - log(x)
    // dF/dx = -1/x
    auto model = AuxiliaryModelBuilder("log_model")
        .state("x")
        .ode("x", modelState("x").log())
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> x = {2.0};
    std::vector<Real> xdot = {0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

    std::vector<Real> dFdx(1);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 1;
    provider.evaluateJacobian(*model, ctx, req);

    // dF/dx = -d(log(x))/dx = -1/x = -0.5
    EXPECT_NEAR(dFdx[0], -0.5, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Sqrt)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = sqrt(x) → F = xdot - sqrt(x)
    // dF/dx = -1 / (2*sqrt(x))
    auto model = AuxiliaryModelBuilder("sqrt_model")
        .state("x")
        .ode("x", modelState("x").sqrt())
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> x = {4.0};
    std::vector<Real> xdot = {0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

    std::vector<Real> dFdx(1);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 1;
    provider.evaluateJacobian(*model, ctx, req);

    // dF/dx = -1/(2*sqrt(4)) = -1/4 = -0.25
    EXPECT_NEAR(dFdx[0], -0.25, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Min_Max)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = min(x, C) with C=5
    // At x=3 < 5: d(min(x,5))/dx = 1  → dF/dx = -1
    // At x=7 > 5: d(min(x,5))/dx = 0  → dF/dx = 0
    auto C = fe::FormExpr::constant(5.0);
    auto model = AuxiliaryModelBuilder("minmax")
        .state("x")
        .ode("x", fe::min(modelState("x"), C))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // Test x < C
    {
        std::vector<Real> x = {3.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);
        EXPECT_NEAR(dFdx[0], -1.0, 1e-10);
    }

    // Test x > C
    {
        std::vector<Real> x = {7.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);
        EXPECT_NEAR(dFdx[0], 0.0, 1e-10);
    }
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Conditional)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = (x > 0) ? -x : x  (damped oscillation piecewise)
    // At x=2: rhs = -2, dF/dx = d(xdot+x)/dx = 1 (ODE form F = xdot-(-x))
    //   → dF/dx = -(d(-x)/dx) = 1
    // At x=-2: rhs = -2, dF/dx = d(xdot-x)/dx = -1 ... wait, let me be precise.
    //
    // rhs = (x > 0) ? -x : x
    // F = xdot - rhs
    // dF/dx = -d(rhs)/dx
    // d(rhs)/dx = (x > 0) ? d(-x)/dx : d(x)/dx = (x>0) ? -1 : 1
    // dF/dx = (x>0) ? 1 : -1
    auto cond = fe::gt(modelState("x"), fe::FormExpr::constant(0.0));
    auto rhs = fe::conditional(cond, -modelState("x"), modelState("x"));
    auto model = AuxiliaryModelBuilder("cond_model")
        .state("x")
        .ode("x", rhs)
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // x > 0 branch
    {
        std::vector<Real> x = {2.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);
        EXPECT_NEAR(dFdx[0], 1.0, 1e-10);
    }

    // x < 0 branch
    {
        std::vector<Real> x = {-2.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);
        EXPECT_NEAR(dFdx[0], -1.0, 1e-10);
    }
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Abs)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = |x| → F = xdot - |x|
    // dF/dx = -sign(x)
    auto model = AuxiliaryModelBuilder("abs_model")
        .state("x")
        .ode("x", modelState("x").abs())
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // x > 0: dF/dx = -1
    {
        std::vector<Real> x = {3.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);
        EXPECT_NEAR(dFdx[0], -1.0, 1e-10);
    }

    // x < 0: dF/dx = +1
    {
        std::vector<Real> x = {-3.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.1;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);
        EXPECT_NEAR(dFdx[0], 1.0, 1e-10);
    }
}

TEST(AuxiliaryDerivativeProvider, Symbolic_MatchesFD_HodgkinHuxleyGating)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // Simplified Hodgkin-Huxley gating variable ODE:
    //   dm/dt = alpha(V)*(1-m) - beta(V)*m
    // With V as an input (constant w.r.t. m), and alpha/beta as simple
    // exp-based rate constants:
    //   alpha = 0.1 * exp(-V/10)     (constant w.r.t. m)
    //   beta  = 0.5 * exp(-V/20)     (constant w.r.t. m)
    //   dm/dt = alpha*(1 - m) - beta*m
    // F = mdot - alpha*(1-m) + beta*m = mdot - alpha + alpha*m + beta*m
    // dF/dm = alpha + beta

    auto V_input = fe::FormExpr::auxiliaryInputRef(0);
    auto alpha = fe::FormExpr::constant(0.1) * (-V_input / fe::FormExpr::constant(10.0)).exp();
    auto beta = fe::FormExpr::constant(0.5) * (-V_input / fe::FormExpr::constant(20.0)).exp();
    auto m = modelState("m");
    auto rhs = alpha * (fe::FormExpr::constant(1.0) - m) - beta * m;

    auto model = AuxiliaryModelBuilder("hh_gate")
        .state("m")
        .input("V")
        .ode("m", rhs)
        .build();

    // Setup symbolic provider.
    AuxiliaryDerivativeProvider sym_provider;
    sym_provider.setup(*model, {});
    EXPECT_EQ(sym_provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // Setup FD provider for comparison.
    AuxiliaryDerivativePolicy fd_policy;
    fd_policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;
    AuxiliaryDerivativeProvider fd_provider;
    fd_provider.setup(*model, fd_policy);

    // Test at several (m, V) points.
    for (Real V_val : {-60.0, -20.0, 0.0, 20.0}) {
        for (Real m_val : {0.0, 0.3, 0.7, 1.0}) {
            std::vector<Real> x = {m_val};
            std::vector<Real> xdot = {0.0};
            std::vector<Real> inputs = {V_val};
            AuxiliaryLocalContext ctx;
            ctx.x = x; ctx.xdot = xdot; ctx.inputs = inputs;
            ctx.time = 0.0; ctx.dt = 0.01;

            std::vector<Real> sym_dFdx(1), fd_dFdx(1);
            AuxiliaryJacobianRequest sym_req, fd_req;
            sym_req.dF_dx = sym_dFdx; sym_req.n = 1;
            fd_req.dF_dx = fd_dFdx; fd_req.n = 1;

            sym_provider.evaluateJacobian(*model, ctx, sym_req);
            fd_provider.evaluateJacobian(*model, ctx, fd_req);

            // Symbolic and FD should agree to FD precision.
            EXPECT_NEAR(sym_dFdx[0], fd_dFdx[0], 1e-5)
                << "V=" << V_val << " m=" << m_val;
        }
    }
}

TEST(AuxiliaryDerivativeProvider, Symbolic_CompoundExpression_DivPowExp)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = x^2 / (1 + exp(-x))    (logistic-growth-like)
    // F = xdot - x^2/(1+exp(-x))
    // We compare symbolic dF/dx against FD at several points.
    auto x = modelState("x");
    auto rhs = x.pow(fe::FormExpr::constant(2.0)) /
               (fe::FormExpr::constant(1.0) + (-x).exp());

    auto model = AuxiliaryModelBuilder("compound")
        .state("x")
        .ode("x", rhs)
        .build();

    AuxiliaryDerivativeProvider sym_provider;
    sym_provider.setup(*model, {});
    EXPECT_EQ(sym_provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    AuxiliaryDerivativePolicy fd_policy;
    fd_policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;
    AuxiliaryDerivativeProvider fd_provider;
    fd_provider.setup(*model, fd_policy);

    for (Real x_val : {0.5, 1.0, 2.0, -1.0, 3.0}) {
        std::vector<Real> xv = {x_val};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = xv; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> sym_dFdx(1), fd_dFdx(1);
        AuxiliaryJacobianRequest sym_req, fd_req;
        sym_req.dF_dx = sym_dFdx; sym_req.n = 1;
        fd_req.dF_dx = fd_dFdx; fd_req.n = 1;

        sym_provider.evaluateJacobian(*model, ctx, sym_req);
        fd_provider.evaluateJacobian(*model, ctx, fd_req);

        EXPECT_NEAR(sym_dFdx[0], fd_dFdx[0], 1e-4)
            << "x=" << x_val;
    }
}

TEST(AuxiliaryDerivativeProvider, Symbolic_MultiState_ExpCoupling)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // 2-state system with exp coupling:
    //   dx/dt = -exp(y) * x
    //   dy/dt = x^2 - y
    // F0 = xdot + exp(y)*x,  F1 = ydot - x^2 + y
    // dF0/dx = exp(y),  dF0/dy = x*exp(y)
    // dF1/dx = -2*x,    dF1/dy = 1
    auto x_expr = modelState("x");
    auto y_expr = modelState("y");

    auto model = AuxiliaryModelBuilder("exp_coupled")
        .state("x")
        .state("y")
        .ode("x", -y_expr.exp() * x_expr)
        .ode("y", x_expr.pow(fe::FormExpr::constant(2.0)) - y_expr)
        .build();

    AuxiliaryDerivativeProvider sym_provider;
    sym_provider.setup(*model, {});
    EXPECT_EQ(sym_provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    AuxiliaryDerivativePolicy fd_policy;
    fd_policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;
    AuxiliaryDerivativeProvider fd_provider;
    fd_provider.setup(*model, fd_policy);

    std::vector<Real> x = {1.5, 0.5};
    std::vector<Real> xdot = {0.0, 0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

    std::vector<Real> sym_J(4), fd_J(4);
    AuxiliaryJacobianRequest sym_req, fd_req;
    sym_req.dF_dx = sym_J; sym_req.n = 2;
    fd_req.dF_dx = fd_J; fd_req.n = 2;

    sym_provider.evaluateJacobian(*model, ctx, sym_req);
    fd_provider.evaluateJacobian(*model, ctx, fd_req);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(sym_J[i], fd_J[i], 1e-4) << "J[" << i << "]";
    }

    // Verify exact values too.
    // dF0/dx = exp(y) = exp(0.5)
    EXPECT_NEAR(sym_J[0], std::exp(0.5), 1e-10);
    // dF0/dy = x * exp(y) = 1.5 * exp(0.5)
    EXPECT_NEAR(sym_J[1], 1.5 * std::exp(0.5), 1e-10);
    // dF1/dx = -2*x = -3.0
    EXPECT_NEAR(sym_J[2], -3.0, 1e-10);
    // dF1/dy = 1.0
    EXPECT_NEAR(sym_J[3], 1.0, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, ArtifactSourceConsistency_NoExpressions)
{
    using namespace svmp::FE::systems;

    // Custom model without residual expressions → FD fallback.
    // Disable analytic override so the provider tries Symbolic, finds no
    // expressions, and falls back to FD.
    ScalarDecayModel custom_model(1.0);

    AuxiliaryDerivativePolicy policy;
    policy.analytic_override_enabled = false;
    AuxiliaryDerivativeProvider provider;
    provider.setup(custom_model, policy);
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::FiniteDifference);
    EXPECT_EQ(provider.artifact().source, AuxiliaryDerivativeSource::FiniteDifference);
}

TEST(AuxiliaryDerivativeProvider, SymbolicJacobian_dFdxdot_MixedRows)
{
    using namespace svmp::FE::systems;

    // Row 0: ODE (dF0/dxdot0 = 1, dF0/dxdot1 = 0)
    // Row 1: algebraic (dF1/dxdot0 = 0, dF1/dxdot1 = 0)
    auto model = AuxiliaryModelBuilder("mixed")
        .state("x", AuxiliaryVariableKind::Differential)
        .state("z", AuxiliaryVariableKind::Algebraic)
        .ode("x", -modelState("x"))
        .algebraic("z", modelState("z") - svmp::FE::forms::FormExpr::constant(1.0))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> x = {1.0, 1.0};
    std::vector<Real> xdot = {0.0, 0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot;

    std::vector<Real> dFdxdot(4);
    AuxiliaryJacobianRequest req;
    req.dF_dxdot = dFdxdot;
    req.want_dF_dxdot = true;
    req.n = 2;
    // Also need dF_dx to avoid assertion; provide a buffer.
    std::vector<Real> dFdx(4);
    req.dF_dx = dFdx;

    provider.evaluateJacobian(*model, ctx, req);

    // Row 0 (ODE): dF/dxdot = [1, 0]
    EXPECT_DOUBLE_EQ(dFdxdot[0], 1.0);
    EXPECT_DOUBLE_EQ(dFdxdot[1], 0.0);
    // Row 1 (algebraic): dF/dxdot = [0, 0]
    EXPECT_DOUBLE_EQ(dFdxdot[2], 0.0);
    EXPECT_DOUBLE_EQ(dFdxdot[3], 0.0);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_ScalarInnerProduct)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = inner(x, param) where inner on scalars = x*param.
    // With param=3:  rhs = 3*x,  F = xdot - 3x,  dF/dx = -3.
    auto x = modelState("x");
    auto p = fe::FormExpr::parameterRef(0);
    auto rhs = x.inner(p);

    auto model = AuxiliaryModelBuilder("inner_model")
        .state("x")
        .param("k")
        .ode("x", rhs)
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> xv = {2.0};
    std::vector<Real> xdot = {0.0};
    std::vector<Real> params = {3.0};
    AuxiliaryLocalContext ctx;
    ctx.x = xv; ctx.xdot = xdot; ctx.params = params;
    ctx.time = 0.0; ctx.dt = 0.01;

    std::vector<Real> dFdx(1);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 1;
    provider.evaluateJacobian(*model, ctx, req);

    EXPECT_NEAR(dFdx[0], -3.0, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_ScalarInverse)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = inv(x) = 1/x → F = xdot - 1/x → dF/dx = 1/x^2
    auto rhs = modelState("x").inv();

    auto model = AuxiliaryModelBuilder("inv_model")
        .state("x")
        .ode("x", rhs)
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> xv = {2.0};
    std::vector<Real> xdot = {0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = xv; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

    std::vector<Real> dFdx(1);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 1;
    provider.evaluateJacobian(*model, ctx, req);

    // dF/dx = -d(1/x)/dx = -(-1/x^2) = 1/x^2 = 0.25
    EXPECT_NEAR(dFdx[0], 0.25, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, FallbackDiagnosticStored)
{
    using namespace svmp::FE::systems;

    // Model with unsupported Gradient op → FD fallback with diagnostic.
    auto model = AuxiliaryModelBuilder("diag_test")
        .state("x")
        .ode("x", modelState("x").grad())
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});

    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::FiniteDifference);
    // The artifact should contain a non-empty fallback reason.
    EXPECT_FALSE(provider.artifact().fallback_reason.empty());
    // Reason should mention the unsupported type.
    EXPECT_NE(provider.artifact().fallback_reason.find("unsupported"),
              std::string::npos);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Power_VariableExponent)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = x^y (two-state system with variable exponent)
    // F0 = xdot - x^y,  F1 = ydot - 0 (y is just a state, trivial ODE)
    // dF0/dx = -d(x^y)/dx = -y * x^(y-1)
    // dF0/dy = -d(x^y)/dy = -x^y * ln(x)
    auto x = modelState("x");
    auto y = modelState("y");

    auto model = AuxiliaryModelBuilder("var_pow")
        .state("x")
        .state("y")
        .ode("x", x.pow(y))
        .ode("y", fe::FormExpr::constant(0.0))
        .build();

    AuxiliaryDerivativeProvider sym_provider;
    sym_provider.setup(*model, {});
    EXPECT_EQ(sym_provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    AuxiliaryDerivativePolicy fd_policy;
    fd_policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;
    AuxiliaryDerivativeProvider fd_provider;
    fd_provider.setup(*model, fd_policy);

    // Test positive base (including zero base with integer exponents).
    for (Real x_val : {0.5, 1.0, 2.0, 3.0}) {
        for (Real y_val : {1.0, 2.0, 3.0}) {
            std::vector<Real> state = {x_val, y_val};
            std::vector<Real> xdot = {0.0, 0.0};
            AuxiliaryLocalContext ctx;
            ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

            std::vector<Real> sym_J(4), fd_J(4);
            AuxiliaryJacobianRequest sym_req, fd_req;
            sym_req.dF_dx = sym_J; sym_req.n = 2;
            fd_req.dF_dx = fd_J; fd_req.n = 2;

            sym_provider.evaluateJacobian(*model, ctx, sym_req);
            fd_provider.evaluateJacobian(*model, ctx, fd_req);

            for (int i = 0; i < 4; ++i) {
                EXPECT_NEAR(sym_J[i], fd_J[i], 1e-4)
                    << "x=" << x_val << " y=" << y_val << " J[" << i << "]";
            }
        }
    }
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Power_VariableExponent_ZeroBase)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // d(x^y)/dx = y * x^(y-1).  At x=0:
    //   y=1: d/dx = 1 * 0^0 = 1     (well-defined)
    //   y=2: d/dx = 2 * 0^1 = 0     (well-defined)
    //   y=3: d/dx = 3 * 0^2 = 0     (well-defined)
    // These are the cases that the old primal==0 guard got wrong.
    auto x = modelState("x");
    auto y = modelState("y");

    auto model = AuxiliaryModelBuilder("zero_base_pow")
        .state("x")
        .state("y")
        .ode("x", x.pow(y))
        .ode("y", fe::FormExpr::constant(0.0))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // x=0, y=1 → d(x^y)/dx = y * x^(y-1) = 1 * 0^0 = 1
    // F = xdot - x^y, dF/dx = -d(x^y)/dx = -1
    {
        std::vector<Real> state = {0.0, 1.0};
        std::vector<Real> xdot = {0.0, 0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> J(4);
        AuxiliaryJacobianRequest req;
        req.dF_dx = J; req.n = 2;
        provider.evaluateJacobian(*model, ctx, req);

        // dF0/dx = -y * x^(y-1) = -1 * 0^0 = -1
        EXPECT_NEAR(J[0], -1.0, 1e-10) << "d(x^1)/dx at x=0 should be 1";
    }

    // x=0, y=2 → d(x^y)/dx = 2 * 0^1 = 0
    {
        std::vector<Real> state = {0.0, 2.0};
        std::vector<Real> xdot = {0.0, 0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> J(4);
        AuxiliaryJacobianRequest req;
        req.dF_dx = J; req.n = 2;
        provider.evaluateJacobian(*model, ctx, req);

        EXPECT_NEAR(J[0], 0.0, 1e-10) << "d(x^2)/dx at x=0 should be 0";
    }

    // x=0, y=3 → d(x^y)/dx = 3 * 0^2 = 0
    {
        std::vector<Real> state = {0.0, 3.0};
        std::vector<Real> xdot = {0.0, 0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> J(4);
        AuxiliaryJacobianRequest req;
        req.dF_dx = J; req.n = 2;
        provider.evaluateJacobian(*model, ctx, req);

        EXPECT_NEAR(J[0], 0.0, 1e-10) << "d(x^3)/dx at x=0 should be 0";
    }
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Power_NegativeBase_IntegerExponent)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = x^3 where x can be negative.
    // dF/dx = -3*x^2 (always non-negative)
    auto model = AuxiliaryModelBuilder("neg_pow")
        .state("x")
        .ode("x", modelState("x").pow(fe::FormExpr::constant(3.0)))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // Negative x: x=-2, dF/dx = -3*(-2)^2 = -12
    {
        std::vector<Real> x = {-2.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);

        EXPECT_NEAR(dFdx[0], -12.0, 1e-10);
    }

    // Zero x: x=0, dF/dx = -3*0^2 = 0
    {
        std::vector<Real> x = {0.0};
        std::vector<Real> xdot = {0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = x; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> dFdx(1);
        AuxiliaryJacobianRequest req;
        req.dF_dx = dFdx; req.n = 1;
        provider.evaluateJacobian(*model, ctx, req);

        EXPECT_NEAR(dFdx[0], 0.0, 1e-10);
    }
}

TEST(AuxiliaryDerivativeProvider, SetupResetsCleansStaleState)
{
    using namespace svmp::FE::systems;

    // First setup: FD fallback (model with unsupported grad op).
    auto model_bad = AuxiliaryModelBuilder("bad")
        .state("x")
        .ode("x", modelState("x").grad())
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model_bad, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::FiniteDifference);
    EXPECT_FALSE(provider.artifact().fallback_reason.empty());

    // Re-setup same provider with a good model → should be clean Symbolic.
    auto model_good = AuxiliaryModelBuilder("good")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    provider.setup(*model_good, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);
    EXPECT_TRUE(provider.hasSymbolicArtifacts());
    EXPECT_TRUE(provider.artifact().fallback_reason.empty());
    EXPECT_EQ(provider.artifact().model_name, "good");
}

TEST(AuxiliaryDerivativeProvider, TensorOpOnNonScalar_FallsBackToFD)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // inner(AsVector(x, x), const_vec) — AsVector is a non-scalar constructor
    // on state-dependent values.  diffWrtAuxSlot should throw on AsVector
    // (default case), causing FD fallback.  This proves the recursive-throw
    // invariant: tensor ops can't silently apply scalar rules to non-scalar
    // operands because the non-scalar constructor throws first.
    auto x = modelState("x");
    auto vec = fe::FormExpr::asVector({x, x});
    auto const_vec = fe::FormExpr::asVector(
        {fe::FormExpr::constant(1.0), fe::FormExpr::constant(1.0)});
    auto rhs = vec.inner(const_vec);

    auto model = AuxiliaryModelBuilder("tensor_inner")
        .state("x")
        .ode("x", rhs)
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});

    // Should fall back to FD because AsVector(state, state) throws
    // during symbolic differentiation.
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::FiniteDifference);
    EXPECT_FALSE(provider.artifact().fallback_reason.empty());
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Power_NegativeBase_VariableExponent_NaNDomain)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // pow(a, b) with a < 0 and non-integer b is NaN in reals.
    // The symbolic derivative should also produce NaN (domain error
    // propagation), matching std::pow and Dual::pow behavior.
    auto x = modelState("x");
    auto y = modelState("y");

    auto model = AuxiliaryModelBuilder("nan_pow")
        .state("x")
        .state("y")
        .ode("x", x.pow(y))
        .ode("y", fe::FormExpr::constant(0.0))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // x = -2, y = 1.5 → pow(-2, 1.5) is NaN → derivative is NaN.
    std::vector<Real> state = {-2.0, 1.5};
    std::vector<Real> xdot = {0.0, 0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

    std::vector<Real> J(4);
    AuxiliaryJacobianRequest req;
    req.dF_dx = J; req.n = 2;
    provider.evaluateJacobian(*model, ctx, req);

    // dF0/dx and dF0/dy should be NaN (domain error propagates).
    EXPECT_TRUE(std::isnan(J[0])) << "dF0/dx should be NaN for pow(-2, 1.5)";
    EXPECT_TRUE(std::isnan(J[1])) << "dF0/dy should be NaN for pow(-2, 1.5)";
    // dF1/dx = 0, dF1/dy = 0 (trivial row).
    EXPECT_NEAR(J[2], 0.0, 1e-10);
    EXPECT_NEAR(J[3], 0.0, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Power_NegativeBase_IntegerVariable_NaNForDb)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // a = -2, b = 3 (variable, happens to be integer).
    // pow(-2, 3) = -8 (well-defined at this point).
    //
    // dF0/dx = -d(x^y)/dx = -y * x^(y-1) = -3 * (-2)^2 = -12.
    //   This is CORRECT — the da term uses a^(b-1) which is fine for
    //   negative a with integer b.
    //
    // dF0/dy = -d(x^y)/dy = -x^y * ln(x) = -(-8) * ln(-2) = NaN.
    //   This is CORRECT — ∂/∂b of a^b at a < 0 does not exist in the
    //   reals because a^b is undefined for non-integer b near b=3.
    //   The symbolic derivative uses ln(a) (not ln(|a|)), producing NaN,
    //   which signals the domain issue.
    auto x = modelState("x");
    auto y = modelState("y");

    auto model = AuxiliaryModelBuilder("neg_int_pow")
        .state("x")
        .state("y")
        .ode("x", x.pow(y))
        .ode("y", fe::FormExpr::constant(0.0))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    std::vector<Real> state = {-2.0, 3.0};
    std::vector<Real> xdot = {0.0, 0.0};
    AuxiliaryLocalContext ctx;
    ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

    std::vector<Real> J(4);
    AuxiliaryJacobianRequest req;
    req.dF_dx = J; req.n = 2;
    provider.evaluateJacobian(*model, ctx, req);

    // dF0/dx: da term = -y * x^(y-1) = -3 * 4 = -12.  Correct.
    EXPECT_NEAR(J[0], -12.0, 1e-10);

    // dF0/dy: db term involves ln(-2) = NaN.
    // ∂/∂b of a^b at a < 0 does not exist in reals.
    EXPECT_TRUE(std::isnan(J[1]))
        << "dF0/dy should be NaN for variable-exponent pow at a=-2, b=3 "
           "(∂/∂b undefined for negative base)";
}

TEST(AuxiliaryDerivativeProvider, Symbolic_Power_ZeroBase_NonPositiveVariableExponent)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // a=0, b=y with variable y.  0^y is only defined for y > 0.
    // At y <= 0, 0^y is undefined → derivative w.r.t. y should be NaN.
    auto x = modelState("x");
    auto y = modelState("y");

    auto model = AuxiliaryModelBuilder("zero_base_neg_exp")
        .state("x")
        .state("y")
        .ode("x", x.pow(y))
        .ode("y", fe::FormExpr::constant(0.0))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    // x=0, y=0: 0^0 is conventionally 1 (std::pow returns 1),
    // but ∂(0^y)/∂y = 0^y * ln(0) which is 1 * (-∞) = -∞.
    // Not well-defined → NaN/Inf is acceptable.
    {
        std::vector<Real> state = {0.0, 0.0};
        std::vector<Real> xdot = {0.0, 0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> J(4);
        AuxiliaryJacobianRequest req;
        req.dF_dx = J; req.n = 2;
        provider.evaluateJacobian(*model, ctx, req);

        // dF0/dy: undefined at a=0, b=0 → must signal domain error.
        EXPECT_FALSE(std::isfinite(J[1]))
            << "dF0/dy at x=0, y=0 should be NaN or Inf (domain error)";
    }

    // x=0, y=-1: 0^(-1) = ∞ (undefined) → derivative NaN/Inf.
    {
        std::vector<Real> state = {0.0, -1.0};
        std::vector<Real> xdot = {0.0, 0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> J(4);
        AuxiliaryJacobianRequest req;
        req.dF_dx = J; req.n = 2;
        provider.evaluateJacobian(*model, ctx, req);

        // dF0/dy: undefined at a=0, b<0 → must signal domain error.
        EXPECT_FALSE(std::isfinite(J[1]))
            << "dF0/dy at x=0, y=-1 should be NaN or Inf (domain error)";
    }

    // x=0, y=2 (positive): db term should be 0 (well-defined case).
    {
        std::vector<Real> state = {0.0, 2.0};
        std::vector<Real> xdot = {0.0, 0.0};
        AuxiliaryLocalContext ctx;
        ctx.x = state; ctx.xdot = xdot; ctx.time = 0.0; ctx.dt = 0.01;

        std::vector<Real> J(4);
        AuxiliaryJacobianRequest req;
        req.dF_dx = J; req.n = 2;
        provider.evaluateJacobian(*model, ctx, req);

        // dF0/dy: 0^2 * ln(0) = 0 * (-∞) → 0 for b > 0. Should be 0.
        EXPECT_NEAR(J[1], 0.0, 1e-10)
            << "dF0/dy at x=0, y=2 should be 0 (well-defined b>0 case)";
    }
}

// ---------------------------------------------------------------------------
//  dF/d(inputs) symbolic sensitivity tests
// ---------------------------------------------------------------------------

TEST(AuxiliaryDerivativeProvider, Symbolic_dFdInputs_LinearModel)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = Q * x → F = xdot - Q*x
    // dF/dQ = -x (for ODE row: dF = -d(rhs)/dQ = -x)
    auto model = AuxiliaryModelBuilder("lin_inp")
        .state("x")
        .input("Q")
        .ode("x", modelInput("Q") * modelState("x"))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, {});
    EXPECT_EQ(provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);
    EXPECT_EQ(provider.artifact().n_inputs, 1);
    EXPECT_FALSE(provider.artifact().dF_dinputs_exprs.empty());

    std::vector<Real> x = {3.0};
    std::vector<Real> xdot = {0.0};
    std::vector<Real> inputs = {2.0};
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.inputs = inputs;
    ctx.time = 0.0; ctx.dt = 0.01;

    std::vector<Real> dFdx(1), dFdinp(1);
    AuxiliaryJacobianRequest req;
    req.dF_dx = dFdx; req.n = 1;
    req.dF_dinputs = dFdinp; req.n_inputs = 1;
    provider.evaluateJacobian(*model, ctx, req);

    // dF/dx = -Q = -2
    EXPECT_NEAR(dFdx[0], -2.0, 1e-10);
    // dF/dQ = -x = -3
    EXPECT_NEAR(dFdinp[0], -3.0, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, Symbolic_dFdInputs_MatchesFD)
{
    using namespace svmp::FE::systems;
    namespace fe = svmp::FE::forms;

    // dx/dt = exp(-Q) * x^2 → F = xdot - exp(-Q)*x^2
    // dF/dQ = exp(-Q)*x^2  (chain: -d(exp(-Q)*x^2)/dQ = exp(-Q)*x^2)
    auto model = AuxiliaryModelBuilder("exp_inp")
        .state("x")
        .input("Q")
        .ode("x", (-modelInput("Q")).exp() * modelState("x").pow(fe::FormExpr::constant(2.0)))
        .build();

    AuxiliaryDerivativeProvider sym_provider;
    sym_provider.setup(*model, {});
    EXPECT_EQ(sym_provider.resolvedSource(), AuxiliaryDerivativeSource::Symbolic);

    AuxiliaryDerivativePolicy fd_policy;
    fd_policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;
    AuxiliaryDerivativeProvider fd_provider;
    fd_provider.setup(*model, fd_policy);

    for (Real x_val : {0.5, 1.0, 2.0}) {
        for (Real q_val : {0.0, 1.0, 2.0}) {
            std::vector<Real> x = {x_val};
            std::vector<Real> xdot = {0.0};
            std::vector<Real> inputs = {q_val};
            AuxiliaryLocalContext ctx;
            ctx.x = x; ctx.xdot = xdot; ctx.inputs = inputs;
            ctx.time = 0.0; ctx.dt = 0.01;

            std::vector<Real> sym_dinp(1), fd_dinp(1);
            std::vector<Real> sym_dx(1), fd_dx(1);
            AuxiliaryJacobianRequest sym_req, fd_req;
            sym_req.dF_dx = sym_dx; sym_req.n = 1;
            sym_req.dF_dinputs = sym_dinp; sym_req.n_inputs = 1;
            fd_req.dF_dx = fd_dx; fd_req.n = 1;
            fd_req.dF_dinputs = fd_dinp; fd_req.n_inputs = 1;

            sym_provider.evaluateJacobian(*model, ctx, sym_req);
            fd_provider.evaluateJacobian(*model, ctx, fd_req);

            EXPECT_NEAR(sym_dinp[0], fd_dinp[0], 1e-4)
                << "x=" << x_val << " Q=" << q_val;
        }
    }
}

// ============================================================================
//  Symbolic Hessian tests
// ============================================================================

TEST(AuxiliaryDerivativeProvider, SymbolicHessian_QuadraticModel)
{
    // dx/dt = -k*x^2.  F = xdot + k*x^2.
    // dF/dx = 2*k*x.  d²F/dx² = 2*k = constant.
    using namespace svmp::FE::systems;

    auto model = AuxiliaryModelBuilder("quad")
        .state("x").param("k")
        .ode("x", -modelParam("k") * modelState("x") * modelState("x"))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    AuxiliaryLocalContext ctx;
    Real x[] = {3.0}, xdot[] = {0.0}, params[] = {2.0};
    ctx.x = x; ctx.xdot = xdot; ctx.params = params;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    // Full Hessian: d²F/(dx dx) = 2*k = 4.0.
    Real hess[1];
    AuxiliaryHessianRequest hreq;
    hreq.mode = AuxiliarySecondDerivativeMode::Hessian;
    hreq.hessian = hess;
    hreq.n = 1;

    provider.evaluateHessian(*model, ctx, hreq);
    EXPECT_NEAR(hess[0], 4.0, 1e-10) << "d²F/dx² for -k*x² should be 2*k";
}

TEST(AuxiliaryDerivativeProvider, SymbolicHessian_TwoState)
{
    // F[0] = xdot[0] + x[0]*x[1],  F[1] = xdot[1] + x[0]^2
    // dF[0]/dx = [x[1], x[0]],  dF[1]/dx = [2*x[0], 0]
    // d²F[0]/(dx0 dx0) = 0, d²F[0]/(dx0 dx1) = 1
    // d²F[0]/(dx1 dx0) = 1, d²F[0]/(dx1 dx1) = 0
    // d²F[1]/(dx0 dx0) = 2, all others = 0
    using namespace svmp::FE::systems;

    auto model = AuxiliaryModelBuilder("twostate")
        .state("x").state("y")
        .ode("x", -modelState("x") * modelState("y"))
        .ode("y", -modelState("x") * modelState("x"))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    AuxiliaryLocalContext ctx;
    Real x[] = {3.0, 5.0}, xdot[] = {0, 0};
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    Real hess[8]; // 2*2*2
    AuxiliaryHessianRequest hreq;
    hreq.mode = AuxiliarySecondDerivativeMode::Hessian;
    hreq.hessian = hess;
    hreq.n = 2;

    provider.evaluateHessian(*model, ctx, hreq);

    // d²F[0]/(dx0 dx0) = 0
    EXPECT_NEAR(hess[0*4 + 0*2 + 0], 0.0, 1e-10);
    // d²F[0]/(dx0 dx1) = 1  (from product rule on x*y)
    EXPECT_NEAR(hess[0*4 + 0*2 + 1], 1.0, 1e-10);
    // d²F[0]/(dx1 dx0) = 1
    EXPECT_NEAR(hess[0*4 + 1*2 + 0], 1.0, 1e-10);
    // d²F[0]/(dx1 dx1) = 0
    EXPECT_NEAR(hess[0*4 + 1*2 + 1], 0.0, 1e-10);
    // d²F[1]/(dx0 dx0) = 2  (from x^2)
    EXPECT_NEAR(hess[1*4 + 0*2 + 0], 2.0, 1e-10);
    // d²F[1]/(dx0 dx1) = 0
    EXPECT_NEAR(hess[1*4 + 0*2 + 1], 0.0, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, SymbolicHVP)
{
    // Same two-state model. HVP with v = [1, 0]:
    // HVP[i,j] = Σ_k d²F[i]/(dx_j dx_k) * v[k]
    // = d²F[i]/(dx_j dx_0) * 1 + d²F[i]/(dx_j dx_1) * 0
    // = d²F[i]/(dx_j dx_0)
    // So HVP = column 0 of each Hessian slice.
    // HVP[0,0] = 0, HVP[0,1] = 1, HVP[1,0] = 2, HVP[1,1] = 0.
    using namespace svmp::FE::systems;

    auto model = AuxiliaryModelBuilder("twostate_hvp")
        .state("x").state("y")
        .ode("x", -modelState("x") * modelState("y"))
        .ode("y", -modelState("x") * modelState("x"))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    AuxiliaryLocalContext ctx;
    Real x[] = {3.0, 5.0}, xdot[] = {0, 0};
    ctx.x = x; ctx.xdot = xdot; ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    Real hvp[4], dir[] = {1.0, 0.0};
    AuxiliaryHessianRequest hreq;
    hreq.mode = AuxiliarySecondDerivativeMode::HessianVectorProduct;
    hreq.hvp = hvp;
    hreq.direction = dir;
    hreq.n = 2;

    provider.evaluateHessian(*model, ctx, hreq);

    EXPECT_NEAR(hvp[0], 0.0, 1e-10);  // d²F[0]/(dx0 dx0)
    EXPECT_NEAR(hvp[1], 1.0, 1e-10);  // d²F[0]/(dx1 dx0)
    EXPECT_NEAR(hvp[2], 2.0, 1e-10);  // d²F[1]/(dx0 dx0)
    EXPECT_NEAR(hvp[3], 0.0, 1e-10);  // d²F[1]/(dx1 dx0)
}

// ============================================================================
//  Direct dF/d(field) tests
// ============================================================================

TEST(AuxiliaryDerivativeProvider, DirectFieldDerivative_Linear)
{
    // Model: dx/dt = -k*x + u_field,  where u_field = DiscreteField(0).
    // F = xdot + k*x - u_field.
    // dF/d(u_field) = -1  (constant).
    using namespace svmp::FE::systems;
    using namespace svmp::FE::forms;

    const auto u_field = FormExpr::discreteField(
        static_cast<svmp::FE::FieldId>(0),
        svmp::FE::spaces::H1Space(svmp::FE::ElementType::Tetra4, 1), "u");

    auto model = AuxiliaryModelBuilder("field_coupled")
        .state("x")
        .param("k")
        .ode("x", -modelParam("k") * modelState("x") + u_field)
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    // Verify the field is detected in referenced_fields.
    const auto& art = provider.artifact();
    ASSERT_FALSE(art.referenced_fields.empty());
    EXPECT_EQ(art.referenced_fields[0], static_cast<svmp::FE::FieldId>(0));

    // Evaluate dF/d(field_value).
    AuxiliaryLocalContext ctx;
    Real x[] = {3.0}, xdot[] = {0.0}, params[] = {2.0};
    ctx.x = x; ctx.xdot = xdot; ctx.params = params;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    auto dF = provider.evaluateFieldDerivative(
        static_cast<svmp::FE::FieldId>(0), ctx);
    ASSERT_EQ(dF.size(), 1u);
    // dF/d(u_field) = -(d(RHS)/d(u_field)) = -(1) = -1 for ODE row.
    EXPECT_NEAR(dF[0], -1.0, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, DirectFieldDerivative_Nonlinear)
{
    // Model: dx/dt = x * u_field^2.
    // F = xdot - x*u^2.
    // dF/du = -(d(x*u^2)/du) = -2*x*u.
    // At x=3, u=2: dF/du = -2*3*2 = -12.
    using namespace svmp::FE::systems;
    using namespace svmp::FE::forms;

    const auto u_field = FormExpr::discreteField(
        static_cast<svmp::FE::FieldId>(0),
        svmp::FE::spaces::H1Space(svmp::FE::ElementType::Tetra4, 1), "u");

    auto model = AuxiliaryModelBuilder("nonlinear_field")
        .state("x")
        .ode("x", modelState("x") * u_field * u_field)
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    const auto& art = provider.artifact();
    ASSERT_FALSE(art.referenced_fields.empty());
    ASSERT_TRUE(art.dF_dfield_exprs.count(static_cast<svmp::FE::FieldId>(0)));

    const auto& dF_exprs = art.dF_dfield_exprs.at(static_cast<svmp::FE::FieldId>(0));
    ASSERT_EQ(dF_exprs.size(), 1u);
    EXPECT_TRUE(dF_exprs[0].isValid());

    // Evaluate numerically with field_values populated.
    const Real x_val = 3.0, u_val = 2.0;
    Real x[] = {x_val}, xdot[] = {0.0};
    svmp::FE::FieldValueEntry fve;
    fve.field = static_cast<svmp::FE::FieldId>(0);
    fve.n_components = 1;
    fve.components[0] = u_val;
    std::vector<svmp::FE::FieldValueEntry> fv = {fve};

    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;
    ctx.field_values = fv;

    auto dF = provider.evaluateFieldDerivative(
        static_cast<svmp::FE::FieldId>(0), ctx);
    ASSERT_EQ(dF.size(), 1u);
    // dF/du = -2*x*u = -2*3*2 = -12
    EXPECT_NEAR(dF[0], -2.0 * x_val * u_val, 1e-10);

    // Verify against finite difference of the residual.
    const Real eps = 1e-7;
    auto evalRes = [&](Real u_pert) {
        // F = xdot - x * u^2.  For xdot=0: F = -x*u^2.
        return -x_val * u_pert * u_pert;
    };
    const Real fd = (evalRes(u_val + eps) - evalRes(u_val - eps)) / (2.0 * eps);
    EXPECT_NEAR(dF[0], fd, 1e-5);
}

TEST(AuxiliaryDerivativeProvider, DirectFieldDerivative_NonlinearMultiField)
{
    // Model: dx/dt = u_a * u_b + x.
    // F = xdot - u_a*u_b - x.
    // dF/d(u_a) = -u_b,  dF/d(u_b) = -u_a.
    // At u_a=3, u_b=5: dF/d(u_a) = -5, dF/d(u_b) = -3.
    using namespace svmp::FE::systems;
    using namespace svmp::FE::forms;

    const auto u_a = FormExpr::discreteField(
        static_cast<svmp::FE::FieldId>(0),
        svmp::FE::spaces::H1Space(svmp::FE::ElementType::Tetra4, 1), "u_a");
    const auto u_b = FormExpr::discreteField(
        static_cast<svmp::FE::FieldId>(1),
        svmp::FE::spaces::H1Space(svmp::FE::ElementType::Tetra4, 1), "u_b");

    auto model = AuxiliaryModelBuilder("multi_field")
        .state("x")
        .ode("x", u_a * u_b + modelState("x"))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    const auto& art = provider.artifact();
    ASSERT_EQ(art.referenced_fields.size(), 2u);

    const Real ua = 3.0, ub = 5.0, x_val = 1.0;
    Real x[] = {x_val}, xdot[] = {0.0};
    svmp::FE::FieldValueEntry fve_a;
    fve_a.field = static_cast<svmp::FE::FieldId>(0);
    fve_a.n_components = 1; fve_a.components[0] = ua;
    svmp::FE::FieldValueEntry fve_b;
    fve_b.field = static_cast<svmp::FE::FieldId>(1);
    fve_b.n_components = 1; fve_b.components[0] = ub;
    std::vector<svmp::FE::FieldValueEntry> fv = {fve_a, fve_b};

    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;
    ctx.field_values = fv;

    // dF/d(u_a) = -u_b = -5
    auto dFa = provider.evaluateFieldDerivative(
        static_cast<svmp::FE::FieldId>(0), ctx);
    ASSERT_EQ(dFa.size(), 1u);
    EXPECT_NEAR(dFa[0], -ub, 1e-10);

    // dF/d(u_b) = -u_a = -3
    auto dFb = provider.evaluateFieldDerivative(
        static_cast<svmp::FE::FieldId>(1), ctx);
    ASSERT_EQ(dFb.size(), 1u);
    EXPECT_NEAR(dFb[0], -ua, 1e-10);
}

TEST(AuxiliaryDerivativeProvider, DirectFieldDerivative_NoFieldRef)
{
    // Model with no FE field references — should have empty referenced_fields.
    using namespace svmp::FE::systems;

    auto model = AuxiliaryModelBuilder("no_field")
        .state("x").param("k")
        .ode("x", -modelParam("k") * modelState("x"))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    const auto& art = provider.artifact();
    EXPECT_TRUE(art.referenced_fields.empty());
    EXPECT_TRUE(art.dF_dfield_exprs.empty());

    auto dF = provider.evaluateFieldDerivative(
        static_cast<svmp::FE::FieldId>(0), {});
    EXPECT_TRUE(dF.empty());
}

TEST(AuxiliaryDerivativeProvider, DirectFieldDerivative_VectorComponent)
{
    // Model: dx/dt = component(u, 0) * component(u, 1)
    // F = xdot - u_0*u_1.
    // dF/d(u_0) = -u_1,  dF/d(u_1) = -u_0,  dF/d(u_2) = 0.
    // At u = {3, 5, 7}: dF/du_0 = -5, dF/du_1 = -3, dF/du_2 = 0.
    using namespace svmp::FE::systems;
    using namespace svmp::FE::forms;

    const auto u_vec = FormExpr::discreteField(
        static_cast<svmp::FE::FieldId>(0),
        FormExprNode::SpaceSignature{
            svmp::FE::spaces::SpaceType::H1,
            svmp::FE::FieldType::Vector,
            svmp::FE::Continuity::C0,
            /*value_dimension=*/3, /*topo_dim=*/3, /*order=*/1,
            svmp::FE::ElementType::Tetra4}, "u");

    auto model = AuxiliaryModelBuilder("vec_comp")
        .state("x")
        .ode("x", component(u_vec, 0) * component(u_vec, 1))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    const auto& art = provider.artifact();
    ASSERT_FALSE(art.referenced_fields.empty());
    auto nc_it = art.dF_dfield_ncomp.find(static_cast<svmp::FE::FieldId>(0));
    ASSERT_NE(nc_it, art.dF_dfield_ncomp.end());
    EXPECT_EQ(nc_it->second, 3);  // 3-component field

    // Evaluate with u = {3, 5, 7}.
    Real x[] = {1.0}, xdot[] = {0.0};
    svmp::FE::FieldValueEntry fve;
    fve.field = static_cast<svmp::FE::FieldId>(0);
    fve.n_components = 3;
    fve.components[0] = 3.0; fve.components[1] = 5.0; fve.components[2] = 7.0;
    std::vector<svmp::FE::FieldValueEntry> fv = {fve};

    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;
    ctx.field_values = fv;

    auto dF = provider.evaluateFieldDerivative(
        static_cast<svmp::FE::FieldId>(0), ctx);
    // n_rows=1, n_comp=3 → 3 values: [dF/du_0, dF/du_1, dF/du_2]
    ASSERT_EQ(dF.size(), 3u);
    EXPECT_NEAR(dF[0], -5.0, 1e-10);  // -u_1
    EXPECT_NEAR(dF[1], -3.0, 1e-10);  // -u_0
    EXPECT_NEAR(dF[2],  0.0, 1e-10);  // zero
}

TEST(AuxiliaryDerivativeProvider, DirectFieldDerivative_InnerProduct)
{
    // Model: dx/dt = inner(u, u) = u_0^2 + u_1^2 + u_2^2.
    // F = xdot - (u_0^2 + u_1^2 + u_2^2).
    // dF/d(u_k) = -2*u_k.
    // At u = {2, 3, 4}: dF/du_0 = -4, dF/du_1 = -6, dF/du_2 = -8.
    using namespace svmp::FE::systems;
    using namespace svmp::FE::forms;

    const auto u_vec = FormExpr::discreteField(
        static_cast<svmp::FE::FieldId>(0),
        FormExprNode::SpaceSignature{
            svmp::FE::spaces::SpaceType::H1,
            svmp::FE::FieldType::Vector,
            svmp::FE::Continuity::C0,
            /*value_dimension=*/3, /*topo_dim=*/3, /*order=*/1,
            svmp::FE::ElementType::Tetra4}, "u");

    auto model = AuxiliaryModelBuilder("inner_vec")
        .state("x")
        .ode("x", inner(u_vec, u_vec))
        .build();

    AuxiliaryDerivativeProvider provider;
    provider.setup(*model, AuxiliaryDerivativePolicy{});

    const auto& art = provider.artifact();
    ASSERT_FALSE(art.referenced_fields.empty());

    Real x[] = {1.0}, xdot[] = {0.0};
    svmp::FE::FieldValueEntry fve;
    fve.field = static_cast<svmp::FE::FieldId>(0);
    fve.n_components = 3;
    fve.components[0] = 2.0; fve.components[1] = 3.0; fve.components[2] = 4.0;
    std::vector<svmp::FE::FieldValueEntry> fv = {fve};

    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;
    ctx.field_values = fv;

    auto dF = provider.evaluateFieldDerivative(
        static_cast<svmp::FE::FieldId>(0), ctx);
    ASSERT_EQ(dF.size(), 3u);
    EXPECT_NEAR(dF[0], -4.0, 1e-10);  // -2*u_0
    EXPECT_NEAR(dF[1], -6.0, 1e-10);  // -2*u_1
    EXPECT_NEAR(dF[2], -8.0, 1e-10);  // -2*u_2
}
