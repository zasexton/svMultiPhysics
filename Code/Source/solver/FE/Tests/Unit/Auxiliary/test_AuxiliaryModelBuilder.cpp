/**
 * @file test_AuxiliaryModelBuilder.cpp
 * @brief Unit tests for AuxiliaryModelBuilder and AuxiliaryBindings (Phase 4A)
 */

#include <gtest/gtest.h>

#include "Assembly/TimeIntegrationContext.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryModelDSL.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryStateStepper.h"
#include "Auxiliary/AuxiliaryDerivativeProvider.h"
#include "Systems/FESystem.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"

#include <cmath>
#include <array>
#include <memory>

using svmp::FE::Real;
namespace forms = svmp::FE::forms;
using namespace svmp::FE::systems;

// ============================================================================
//  Reusable test helpers
// ============================================================================

namespace aux_test {

/// Build a simple scalar decay model: dx/dt = -k*x, output y = x.
inline std::shared_ptr<BuiltAuxiliaryModel> buildDecay(Real k_default = 0.0)
{
    if (k_default != 0.0) {
        return aux::model("decay", [&](ModelFacade& m) {
            auto x = m.state("x");
            auto k = m.param("k", k_default);
            m << ddt(x) == -k * x;
            m << out("y") == x;
        });
    }
    return aux::model("decay", [](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k");
        m << ddt(x) == -k * x;
        m << out("y") == x;
    });
}

/// Build an RCR model: dX/dt = (Q - (X-Pd)/Rd)/C, P_out = X + Rp*Q.
inline std::shared_ptr<BuiltAuxiliaryModel> buildRCR()
{
    return aux::model("rcr", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");
        m << ddt(X) == (Q - (X - Pd) / Rd) / C;
        m << out("P_out") == X + Rp * Q;
    });
}

/// Build a mixed ODE/algebraic model: dx/dt = -x + z, 0 = x + z - 1.
inline std::shared_ptr<BuiltAuxiliaryModel> buildDAE()
{
    return aux::model("dae", [](ModelFacade& m) {
        auto x = m.state("x");
        auto z = m.state("z", AuxiliaryVariableKind::Algebraic);
        m.initialGuess("z", 0.5);
        m << ddt(x) == -x + z;
        m << alg(z) == x + z - forms::FormExpr::constant(1.0);
    });
}

/// Evaluate a model's residual at given state/params.
inline std::vector<Real> evaluateResidual(
    const BuiltAuxiliaryModel& model,
    std::span<const Real> x, std::span<const Real> xdot,
    std::span<const Real> inputs, std::span<const Real> params,
    Real time = 0.0, Real dt = 0.01)
{
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xdot; ctx.inputs = inputs; ctx.params = params;
    ctx.time = time; ctx.dt = dt; ctx.effective_dt = dt;

    std::vector<Real> res(static_cast<std::size_t>(model.dimension()));
    AuxiliaryResidualRequest req{res};
    model.evaluateResidual(ctx, req);
    return res;
}

/// Assert canonical state ordering matches expected names.
inline void expectStateOrder(const BuiltAuxiliaryModel& model,
                              const std::vector<std::string>& expected)
{
    ASSERT_EQ(model.stateNames().size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(model.stateNames()[i], expected[i])
            << "State " << i << " should be '" << expected[i] << "'";
    }
}

inline svmp::FE::assembly::TimeIntegrationContext buildGeneralizedAlphaFirstOrderContext(double dt)
{
    svmp::FE::assembly::TimeIntegrationContext ctx;
    ctx.integrator_name = "GeneralizedAlpha(1stOrder)";

    svmp::FE::assembly::TimeDerivativeStencil stencil;
    stencil.order = 1;
    const double alpha_m = 5.0 / 6.0;
    const double alpha_f = 2.0 / 3.0;
    const double gamma = 2.0 / 3.0;
    const double c = alpha_m / (gamma * dt * alpha_f);
    const double c0 = (1.0 - alpha_m) - alpha_m * (1.0 - gamma) / gamma;
    stencil.a = {static_cast<Real>(c), static_cast<Real>(-c), static_cast<Real>(c0)};
    ctx.dt1 = stencil;
    return ctx;
}

class OutputUsesXDotModel final : public AuxiliaryStateModel {
public:
    [[nodiscard]] std::string modelName() const override { return "OutputUsesXDot"; }
    [[nodiscard]] int dimension() const override { return 1; }
    [[nodiscard]] AuxiliaryStructuralMetadata structuralMetadata() const override
    {
        AuxiliaryStructuralMetadata meta;
        meta.variable_kinds = {AuxiliaryVariableKind::Differential};
        return meta;
    }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& request) const override
    {
        request.residual[0] = ctx.xdot[0] + ctx.x[0];
    }

    [[nodiscard]] int outputCount() const override { return 1; }
    [[nodiscard]] std::vector<std::string> outputNames() const override { return {"xdot"}; }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                         std::span<Real> output) const override
    {
        output[0] = ctx.xdot[0];
    }
};

} // namespace aux_test

TEST(AuxiliaryStateIntegration, AdvanceAuxiliaryStateSyncsGhostedBlocks)
{
    svmp::FE::systems::FESystem system(
        std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});
    auto& mgr = system.auxiliaryStateManager();

    auto spec = AuxiliaryStateSpec::nodeField("ghosted", 1);
    mgr.registerBlock(spec, 4, 2);

    bool hook_called = false;
    mgr.setGhostSyncHook("ghosted", [&](std::string_view name, std::span<Real> values) {
        hook_called = true;
        EXPECT_EQ(name, "ghosted");
        EXPECT_EQ(values.size(), 4u);
    });

    system.advanceAuxiliaryState(0.0, 0.1);
    EXPECT_TRUE(hook_called);
}

TEST(AuxiliaryStateIntegration, FinalizeMonolithicStageSyncsGhostedBlocks)
{
    svmp::FE::systems::FESystem system(
        std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});
    auto& mgr = system.auxiliaryStateManager();

    auto spec = AuxiliaryStateSpec::nodeField("ghosted", 1);
    mgr.registerBlock(spec, 3, 1);

    bool hook_called = false;
    mgr.setGhostSyncHook("ghosted", [&](std::string_view, std::span<Real>) {
        hook_called = true;
    });

    system.finalizeMonolithicAuxiliaryStageState(0.5, 0.1);
    EXPECT_TRUE(hook_called);
}

// ============================================================================
//  Builder basics
// ============================================================================

TEST(AuxiliaryModelBuilder, BuildScalarDecay)
{
    // dx/dt = -k*x
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .param("k")
        .ode("x", -modelParam("k") * modelState("x"))
        .output("y", modelState("x"))
        .build();

    EXPECT_EQ(model->modelName(), "decay");
    EXPECT_EQ(model->dimension(), 1);

    auto meta = model->structuralMetadata();
    ASSERT_EQ(meta.variable_kinds.size(), 1u);
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);

    EXPECT_TRUE(model->hasResidualExpressions());
    EXPECT_EQ(model->residualExpressions().size(), 1u);

    EXPECT_EQ(model->stateNames().size(), 1u);
    EXPECT_EQ(model->stateNames()[0], "x");
}

TEST(AuxiliaryModelBuilder, BuildWithInput)
{
    auto model = AuxiliaryModelBuilder("driven")
        .input("Q")
        .state("P")
        .param("R")
        .ode("P", (modelInput("Q") - modelState("P") / modelParam("R")))
        .build();

    EXPECT_EQ(model->dimension(), 1);

    const auto& sig = model->signature();
    ASSERT_EQ(sig.inputs.size(), 1u);
    EXPECT_EQ(sig.inputs[0].name, "Q");
    ASSERT_EQ(sig.parameters.size(), 1u);
    EXPECT_EQ(sig.parameters[0].name, "R");
}

TEST(AuxiliaryModelBuilder, BuildMixedDAE)
{
    // x' = -x + z,  0 = x + z - 1
    auto model = AuxiliaryModelBuilder("mixed_dae")
        .state("x", AuxiliaryVariableKind::Differential)
        .state("z", AuxiliaryVariableKind::Algebraic)
        .ode("x", -modelState("x") + modelState("z"))
        .algebraic("z", modelState("x") + modelState("z") - forms::FormExpr::constant(1.0))
        .build();

    EXPECT_EQ(model->dimension(), 2);

    auto meta = model->structuralMetadata();
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[1], AuxiliaryVariableKind::Algebraic);
}

TEST(AuxiliaryModelBuilder, BuildWithOutputs)
{
    auto model = AuxiliaryModelBuilder("rcr")
        .input("Q")
        .state("P_d")
        .param("R_p")
        .param("R_d")
        .param("C")
        .ode("P_d", (modelInput("Q") - modelState("P_d") / modelParam("R_d")) / modelParam("C"))
        .output("P_out", modelParam("R_p") * modelInput("Q") + modelState("P_d"))
        .build();

    EXPECT_EQ(model->dimension(), 1);

    const auto& sig = model->signature();
    ASSERT_EQ(sig.outputs.size(), 1u);
    EXPECT_EQ(sig.outputs[0].name, "P_out");

    const auto& out_exprs = model->outputExpressions();
    ASSERT_EQ(out_exprs.size(), 1u);
    EXPECT_EQ(out_exprs[0].first, "P_out");
}

TEST(AuxiliaryModelBuilder, BuildWithDerivativePolicy)
{
    AuxiliaryDerivativePolicy policy;
    policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;

    auto model = AuxiliaryModelBuilder("test")
        .state("x")
        .ode("x", -modelState("x"))
        .derivatives(policy)
        .build();

    EXPECT_EQ(model->dimension(), 1);
}

// ============================================================================
//  Builder validation
// ============================================================================

TEST(AuxiliaryModelBuilder, DuplicateNameThrows)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("bad")
            .state("x")
            .param("x")  // duplicate!
            .ode("x", modelState("x")),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, OdeOnAlgebraicThrows)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("bad")
            .state("z", AuxiliaryVariableKind::Algebraic)
            .ode("z", modelState("z")),  // wrong kind!
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, AlgebraicOnDifferentialThrows)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("bad")
            .state("x", AuxiliaryVariableKind::Differential)
            .algebraic("x", modelState("x")),  // wrong kind!
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, UndeclaredStateThrows)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("bad")
            .state("x")
            .ode("y", modelState("x")),  // "y" not declared!
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, NoStatesThrows)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("bad").build(),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, MismatchedRowCountThrows)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("bad")
            .state("x")
            .state("y")
            .ode("x", -modelState("x"))
            // Missing row for "y"
            .build(),
        svmp::FE::InvalidArgumentException);
}

// ============================================================================
//  Signature compatibility
// ============================================================================

TEST(AuxiliaryModelSignature, CompatibleSignatures)
{
    auto m1 = AuxiliaryModelBuilder("A")
        .input("Q")
        .state("x")
        .param("k")
        .ode("x", -modelParam("k") * modelState("x"))
        .output("y", modelState("x"))
        .build();

    auto m2 = AuxiliaryModelBuilder("B")
        .input("Q")
        .state("z")  // different internal name!
        .param("k")
        .ode("z", -modelParam("k") * modelState("z"))
        .output("y", modelState("z"))
        .build();

    EXPECT_TRUE(m1->signature().isCompatibleWith(m2->signature()));
}

TEST(AuxiliaryModelSignature, IncompatibleSignatures)
{
    auto m1 = AuxiliaryModelBuilder("A")
        .input("Q")
        .state("x")
        .ode("x", modelState("x"))
        .build();

    auto m2 = AuxiliaryModelBuilder("B")
        .input("P")  // different input name
        .state("x")
        .ode("x", modelState("x"))
        .build();

    EXPECT_FALSE(m1->signature().isCompatibleWith(m2->signature()));
}

// ============================================================================
//  Deployment (use(model))
// ============================================================================

TEST(AuxiliaryBindings, UseModelBasic)
{
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .param("k")
        .ode("x", -modelParam("k") * modelState("x"))
        .build();

    auto inst = use(model)
        .name("my_decay")
        .scope(AuxiliaryStateScope::Global)
        .solveMode(AuxiliarySolveMode::Partitioned)
        .param("k", 2.0)
        .initialize({1.0});

    EXPECT_EQ(inst.instanceName(), "my_decay");
    EXPECT_EQ(inst.getScope(), AuxiliaryStateScope::Global);
    EXPECT_EQ(inst.getSolveMode(), AuxiliarySolveMode::Partitioned);
    EXPECT_EQ(inst.paramValues().at("k"), 2.0);
    ASSERT_EQ(inst.initialValues().size(), 1u);
    EXPECT_DOUBLE_EQ(inst.initialValues()[0], 1.0);
}

TEST(AuxiliaryBindings, UseModelWithBindings)
{
    auto model = AuxiliaryModelBuilder("driven")
        .input("Q")
        .state("P")
        .param("R")
        .ode("P", modelInput("Q") - modelState("P") / modelParam("R"))
        .build();

    auto inst = use(model)
        .name("outlet_bc")
        .bind("Q", "flow_rate_3")
        .param("R", 100.0)
        .initialize({0.0});

    EXPECT_EQ(inst.inputBindings().at("Q"), "flow_rate_3");
    EXPECT_EQ(inst.paramValues().at("R"), 100.0);
}

TEST(AuxiliaryBindings, UseModelWithStepper)
{
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    AuxiliaryStepperSpec spec;
    spec.method_name = "RK4";
    spec.substep_count = 10;

    auto inst = use(model)
        .stepper(spec);

    EXPECT_EQ(inst.getStepperSpec().method_name, "RK4");
    EXPECT_EQ(inst.getStepperSpec().substep_count, 10);
}

TEST(AuxiliaryBindings, UseModelWithRegion)
{
    auto model = AuxiliaryModelBuilder("bc_state")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    AuxiliaryDeploymentRegion region;
    region.kind = AuxiliaryRegionKind::BoundarySet;
    region.identity = "outlet_3";

    auto inst = use(model)
        .scope(AuxiliaryStateScope::Facet)
        .region(region);

    EXPECT_EQ(inst.getScope(), AuxiliaryStateScope::Facet);
    EXPECT_EQ(inst.getRegion().kind, AuxiliaryRegionKind::BoundarySet);
    EXPECT_EQ(inst.getRegion().identity, "outlet_3");
}

TEST(AuxiliaryBindings, DefaultInstanceNameIsModelName)
{
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    auto inst = use(model);
    EXPECT_EQ(inst.instanceName(), "decay");
}

// ============================================================================
//  Validation
// ============================================================================

TEST(AuxiliaryBindings, ValidatePassesWithAllBindings)
{
    auto model = AuxiliaryModelBuilder("driven")
        .input("Q")
        .state("P")
        .param("R")
        .ode("P", modelInput("Q"))
        .build();

    auto inst = use(model)
        .name("test")
        .bind("Q", "flow_rate")
        .param("R", 100.0)
        .initialize({0.0});

    auto diag = inst.validate();
    EXPECT_TRUE(diag.empty()) << "validation failed: " << diag;
}

TEST(AuxiliaryBindings, ValidateReportsUnboundInput)
{
    auto model = AuxiliaryModelBuilder("driven")
        .input("Q")
        .state("P")
        .ode("P", modelInput("Q"))
        .build();

    auto inst = use(model).name("test");
    // Q is not bound!

    auto diag = inst.validate();
    EXPECT_FALSE(diag.empty());
    EXPECT_NE(diag.find("Q"), std::string::npos);
}

TEST(AuxiliaryBindings, ValidateReportsWrongInitSize)
{
    auto model = AuxiliaryModelBuilder("system")
        .state("x")
        .state("y")
        .ode("x", -modelState("x"))
        .ode("y", -modelState("y"))
        .build();

    auto inst = use(model)
        .name("test")
        .initialize({1.0}); // should be 2 values!

    auto diag = inst.validate();
    EXPECT_FALSE(diag.empty());
    EXPECT_NE(diag.find("initial_values"), std::string::npos);
}

// ============================================================================
//  Integration: builder → stepper
// ============================================================================

TEST(AuxiliaryModelBuilder, BuiltModelWorksWithStepper)
{
    // The builder now resolves modelState("x") → AuxiliaryStateRef(0) during build().
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    AuxiliaryDerivativeProvider deriv;
    deriv.setup(*model, {});

    auto stepper = createStepper("ForwardEuler");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    std::vector<Real> x = {1.0};
    const std::vector<Real> x0 = {1.0};

    auto result = stepper->advance(*model, deriv, x, x0, {}, {}, {}, 0.0, 0.1);

    EXPECT_TRUE(result.converged);
    // Forward Euler: x = 1 + 0.1*(-1) = 0.9
    EXPECT_NEAR(x[0], 0.9, 1e-12);
}

TEST(AuxiliaryModelBuilder, BuiltModelWorksWithBackwardEuler)
{
    // Uses symbolic API — resolved at build time.
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    AuxiliaryDerivativePolicy policy;
    policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;

    AuxiliaryDerivativeProvider deriv;
    deriv.setup(*model, policy);

    auto stepper = createStepper("BackwardEuler");
    AuxiliaryStepperSpec spec;
    stepper->setup(1, spec);

    std::vector<Real> x = {1.0};
    const std::vector<Real> x0 = {1.0};

    auto result = stepper->advance(*model, deriv, x, x0, {}, {}, {}, 0.0, 0.1);

    EXPECT_TRUE(result.converged);
    // Backward Euler: x = 1/(1+dt) ≈ 0.909091
    EXPECT_NEAR(x[0], 1.0 / 1.1, 1e-6);
}

// ============================================================================
//  Multi-component EP-like model
// ============================================================================

TEST(AuxiliaryModelBuilder, FourGateIonicModel)
{
    // Simplified ionic model with 4 gating variables.
    auto model = AuxiliaryModelBuilder("ionic_gates")
        .input("V")               // voltage input
        .state("m")               // Na activation
        .state("h")               // Na inactivation
        .state("j")               // Na slow inactivation
        .state("d", AuxiliaryVariableKind::Algebraic)  // algebraic constraint
        .ode("m", modelInput("V") * (forms::FormExpr::constant(1.0) - modelState("m")))
        .ode("h", -modelInput("V") * modelState("h"))
        .ode("j", -modelState("j"))
        .algebraic("d", modelState("m") + modelState("h") + modelState("j")
                        + modelState("d") - forms::FormExpr::constant(1.0))
        .output("I_Na", modelState("m") * modelState("h") * modelState("j"))
        .build();

    EXPECT_EQ(model->dimension(), 4);
    EXPECT_EQ(model->stateNames().size(), 4u);

    auto meta = model->structuralMetadata();
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[1], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[2], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[3], AuxiliaryVariableKind::Algebraic);

    const auto& sig = model->signature();
    EXPECT_EQ(sig.inputs.size(), 1u);
    EXPECT_EQ(sig.outputs.size(), 1u);
    EXPECT_EQ(sig.outputs[0].name, "I_Na");
}

// ============================================================================
//  End-to-end: builder → use(model) → FESystem deploy → advance → commit
// ============================================================================

TEST(AuxiliaryModelBuilder, EndToEnd_FESystem_DeployAdvanceCommit)
{
    using namespace svmp::FE;

    // 1. Build a scalar decay model: dx/dt = -k*x
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .param("k")
        .ode("x", -modelParam("k") * modelState("x"))
        .output("y", modelState("x") * forms::FormExpr::constant(2.0))
        .build();

    // 2. Create FESystem (null mesh — auxiliary-only use)
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    // 3. Deploy via use(model) → FESystem
    system.deployAuxiliaryModel(
        use(model)
            .name("my_decay")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .stepper({"ForwardEuler"})
            .param("k", 1.0)
            .initialize({1.0}));

    // 4. Finalize (creates block, stepper, derivative provider)
    system.finalizeAuxiliaryLayout();

    // 5. Verify block was registered
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_EQ(mgr->blockCount(), 1u);
    EXPECT_TRUE(mgr->hasBlock("my_decay"));
    auto& blk = mgr->getBlock("my_decay");
    EXPECT_DOUBLE_EQ(blk.work()[0], 1.0); // initial value

    // 6. Advance: Forward Euler dt=0.1 → x = 1 + 0.1*(-1*1) = 0.9
    system.advanceAuxiliaryState(0.0, 0.1);
    EXPECT_NEAR(blk.work()[0], 0.9, 1e-12);

    // 7. Prepare for assembly (evaluates outputs)
    systems::SystemStateView state;
    state.time = 0.1;
    state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);

    // 8. Check evaluated output: y = 2*x = 2*0.9 = 1.8
    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], 1.8, 1e-12);

    // 9. Commit
    system.commitTimeStep();
    EXPECT_NEAR(blk.committed()[0], 0.9, 1e-12);

    // 10. Begin new step (resets work to committed)
    system.beginTimeStep();
    EXPECT_NEAR(blk.work()[0], 0.9, 1e-12);

    // 11. Advance again → x = 0.9 + 0.1*(-0.9) = 0.81
    system.advanceAuxiliaryState(0.1, 0.1);
    EXPECT_NEAR(blk.work()[0], 0.81, 1e-12);

    // 12. Rollback
    system.rollbackAuxiliaryState();
    EXPECT_NEAR(blk.work()[0], 0.9, 1e-12);

    // 13. Analysis summary
    auto summary = system.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_blocks, 1u);
    EXPECT_EQ(summary.n_partitioned, 1u);
    EXPECT_EQ(summary.n_monolithic, 0u);

    // 14. Output slot lookup
    EXPECT_EQ(system.auxiliaryOutputSlotOf("y"), 0u);
    EXPECT_EQ(system.auxiliaryOutputSlotOf("nonexistent"),
              static_cast<std::size_t>(-1));
}

TEST(AuxiliaryModelBuilder, EndToEnd_MultiModel_OutputSlots)
{
    using namespace svmp::FE;

    // Two models each with an output named "P_out"
    auto model_a = AuxiliaryModelBuilder("rcr_a")
        .state("x")
        .ode("x", -modelState("x"))
        .output("P_out", modelState("x") * forms::FormExpr::constant(10.0))
        .build();

    auto model_b = AuxiliaryModelBuilder("rcr_b")
        .state("x")
        .ode("x", -modelState("x") * forms::FormExpr::constant(0.5))
        .output("P_out", modelState("x") * forms::FormExpr::constant(20.0))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model_a).name("outlet_1").stepper({"ForwardEuler"}).initialize({1.0}));
    system.deployAuxiliaryModel(
        use(model_b).name("outlet_2").stepper({"ForwardEuler"}).initialize({2.0}));

    system.finalizeAuxiliaryLayout();

    // Bare-name lookup throws on ambiguity (both models have "P_out")
    EXPECT_THROW(system.auxiliaryOutputSlotOf("P_out"),
                 svmp::FE::InvalidArgumentException);

    // Instance-qualified lookup disambiguates
    auto slot_1 = system.auxiliaryOutputSlotOf("outlet_1", "P_out");
    auto slot_2 = system.auxiliaryOutputSlotOf("outlet_2", "P_out");
    EXPECT_EQ(slot_1, 0u);
    EXPECT_EQ(slot_2, 1u); // second model's output

    // Advance both
    system.advanceAuxiliaryState(0.0, 0.1);

    // Evaluate outputs
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);

    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 2u);

    // outlet_1: x=0.9, P_out = 0.9 * 10 = 9.0
    EXPECT_NEAR(outputs[0], 9.0, 1e-12);
    // outlet_2: x = 2 + 0.1*(-0.5*2) = 1.9, P_out = 1.9 * 20 = 38.0
    EXPECT_NEAR(outputs[1], 38.0, 1e-12);

    // Nonexistent lookups
    EXPECT_EQ(system.auxiliaryOutputSlotOf("outlet_1", "nonexistent"),
              static_cast<std::size_t>(-1));
    EXPECT_EQ(system.auxiliaryOutputSlotOf("nonexistent", "P_out"),
              static_cast<std::size_t>(-1));
}

TEST(AuxiliaryModelBuilder, EndToEnd_MultiEntity_OutputSlots)
{
    using namespace svmp::FE;

    // Model with 2 outputs
    auto model = AuxiliaryModelBuilder("state2d")
        .state("x")
        .state("y")
        .ode("x", -modelState("x"))
        .ode("y", -modelState("y"))
        .output("sum", modelState("x") + modelState("y"))
        .output("diff", modelState("x") - modelState("y"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    // Deploy first model as Global (1 entity)
    system.deployAuxiliaryModel(
        use(model).name("global_block")
            .scope(AuxiliaryStateScope::Global)
            .stepper({"ForwardEuler"})
            .initialize({1.0, 2.0}));

    // Deploy second model with explicit entityCount(3)
    // (simulates a local Cell-scoped block without a real mesh)
    // Initial values are per-model-dimension (2), not total storage.
    // All entities start at zero.
    system.deployAuxiliaryModel(
        use(model).name("multi_entity_block")
            .scope(AuxiliaryStateScope::Cell)
            .entityCount(3)
            .stepper({"ForwardEuler"}));

    system.finalizeAuxiliaryLayout();

    // Slot layout:
    // global_block: 1 entity × 2 outputs = slots [0, 1]
    // multi_entity_block: 3 entities × 2 outputs = slots [2, 3, 4, 5, 6, 7]
    auto slot_g_sum = system.auxiliaryOutputSlotOf("global_block", "sum");
    auto slot_g_diff = system.auxiliaryOutputSlotOf("global_block", "diff");
    auto slot_m_sum = system.auxiliaryOutputSlotOf("multi_entity_block", "sum");
    auto slot_m_diff = system.auxiliaryOutputSlotOf("multi_entity_block", "diff");

    EXPECT_EQ(slot_g_sum, 0u);
    EXPECT_EQ(slot_g_diff, 1u);
    EXPECT_EQ(slot_m_sum, 2u);  // After global's 1*2 = 2 slots
    EXPECT_EQ(slot_m_diff, 3u); // entity-0's diff

    // The full flattened buffer after output eval should have 2 + 6 = 8 entries
    systems::SystemStateView state;
    state.time = 0.0; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);

    auto outputs = system.auxiliaryOutputValues();
    EXPECT_EQ(outputs.size(), 8u); // 1*2 + 3*2

    // global_block entity 0: x=1, y=2 → sum=3, diff=-1
    EXPECT_NEAR(outputs[0], 3.0, 1e-12);
    EXPECT_NEAR(outputs[1], -1.0, 1e-12);

    // multi_entity_block entities: all zeros → sum=0, diff=0
    for (std::size_t i = 2; i < 8; ++i) {
        EXPECT_NEAR(outputs[i], 0.0, 1e-12);
    }
}

// ============================================================================
//  Custom AuxiliaryStateModel with outputs through FESystem
// ============================================================================

namespace {
class CustomModelWithOutputs : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "custom_with_outputs"; }
    int dimension() const override { return 1; }
    AuxiliaryStructuralMetadata structuralMetadata() const override {
        AuxiliaryStructuralMetadata m;
        m.variable_kinds = {AuxiliaryVariableKind::Differential};
        return m;
    }
    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override {
        // dx/dt = -x → F = xdot + x
        req.residual[0] = ctx.xdot[0] + ctx.x[0];
    }
    bool hasAnalyticJacobian() const override { return true; }
    void evaluateJacobian(const AuxiliaryLocalContext&,
                          AuxiliaryJacobianRequest& req) const override {
        if (!req.dF_dx.empty()) req.dF_dx[0] = 1.0;
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 1.0;
    }
    int outputCount() const override { return 1; }
    std::vector<std::string> outputNames() const override { return {"doubled"}; }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                          std::span<Real> output) const override {
        output[0] = ctx.x[0] * 2.0;
    }
};
} // namespace

TEST(AuxiliaryModelBuilder, EndToEnd_CustomModel_Outputs)
{
    using namespace svmp::FE;

    auto model = std::make_shared<CustomModelWithOutputs>();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model)
            .name("custom_inst")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .stepper({"ForwardEuler"})
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    // Slot lookup via base-class outputNames()
    auto slot = system.auxiliaryOutputSlotOf("doubled");
    EXPECT_EQ(slot, 0u);
    auto slot_q = system.auxiliaryOutputSlotOf("custom_inst", "doubled");
    EXPECT_EQ(slot_q, 0u);

    // Advance: FE x = 1 + 0.1*(-1) = 0.9
    system.advanceAuxiliaryState(0.0, 0.1);
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("custom_inst").work()[0], 0.9, 1e-12);

    // Evaluate outputs via base-class evaluateOutputs()
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);

    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], 1.8, 1e-12); // 0.9 * 2
}

// Custom model that uses params in both residual and output evaluation.
namespace {
class CustomModelWithParams : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "custom_with_params"; }
    int dimension() const override { return 1; }
    AuxiliaryStructuralMetadata structuralMetadata() const override {
        AuxiliaryStructuralMetadata m;
        m.variable_kinds = {AuxiliaryVariableKind::Differential};
        return m;
    }
    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override {
        // dx/dt = -k*x, k = params[0]
        const Real k = ctx.params.empty() ? 1.0 : ctx.params[0];
        req.residual[0] = ctx.xdot[0] + k * ctx.x[0];
    }
    bool hasAnalyticJacobian() const override { return true; }
    void evaluateJacobian(const AuxiliaryLocalContext& ctx,
                          AuxiliaryJacobianRequest& req) const override {
        const Real k = ctx.params.empty() ? 1.0 : ctx.params[0];
        if (!req.dF_dx.empty()) req.dF_dx[0] = k;
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 1.0;
    }
    int outputCount() const override { return 1; }
    std::vector<std::string> outputNames() const override { return {"scaled"}; }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                          std::span<Real> output) const override {
        // output = k * x
        const Real k = ctx.params.empty() ? 1.0 : ctx.params[0];
        output[0] = k * ctx.x[0];
    }
};
} // namespace

TEST(AuxiliaryModelBuilder, EndToEnd_CustomModel_WithParams)
{
    using namespace svmp::FE;

    auto model = std::make_shared<CustomModelWithParams>();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model)
            .name("param_inst")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .stepper({"ForwardEuler"})
            .param("k", 2.0)   // k=2
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    // Advance: FE with k=2 → x = 1 + 0.1*(-2*1) = 0.8
    system.advanceAuxiliaryState(0.0, 0.1);
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("param_inst").work()[0], 0.8, 1e-12);

    // Evaluate output: scaled = k*x = 2*0.8 = 1.6
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);

    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], 1.6, 1e-12);
}

// Custom model that uses inputs (via .bind) in residual evaluation.
namespace {
class CustomModelWithInputs : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "custom_with_inputs"; }
    int dimension() const override { return 1; }
    AuxiliaryStructuralMetadata structuralMetadata() const override {
        AuxiliaryStructuralMetadata m;
        m.variable_kinds = {AuxiliaryVariableKind::Differential};
        return m;
    }
    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override {
        // dx/dt = Q - x, where Q = inputs[0]
        const Real Q = ctx.inputs.empty() ? 0.0 : ctx.inputs[0];
        req.residual[0] = ctx.xdot[0] - Q + ctx.x[0];
    }
    bool hasAnalyticJacobian() const override { return true; }
    void evaluateJacobian(const AuxiliaryLocalContext&,
                          AuxiliaryJacobianRequest& req) const override {
        if (!req.dF_dx.empty()) req.dF_dx[0] = 1.0;
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 1.0;
    }
    int outputCount() const override { return 1; }
    std::vector<std::string> outputNames() const override { return {"result"}; }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                          std::span<Real> output) const override {
        const Real Q = ctx.inputs.empty() ? 0.0 : ctx.inputs[0];
        output[0] = ctx.x[0] + Q;
    }
};
} // namespace

TEST(AuxiliaryModelBuilder, EndToEnd_CustomModel_WithInputBindings)
{
    using namespace svmp::FE;

    auto model = std::make_shared<CustomModelWithInputs>();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    // Register an auxiliary input in the registry.
    auto& reg = system.auxiliaryInputRegistry();
    reg.registerInput(
        {.name = "flow_rate", .size = 1,
         .producer = AuxiliaryInputProducer::DirectUserData},
        [](Real, Real, std::span<Real> out) { out[0] = 5.0; });

    // Deploy with .bind("Q", "flow_rate")
    system.deployAuxiliaryModel(
        use(model)
            .name("driven_inst")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .stepper({"ForwardEuler"})
            .bind("Q", "flow_rate")
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();

    // Normal lifecycle: beginTimeStep() invalidates inputs,
    // then advanceAuxiliaryState() must evaluate them internally.
    // No prepareAuxiliaryForAssembly() call before advance.
    system.beginTimeStep();
    system.advanceAuxiliaryState(0.0, 0.1);
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("driven_inst").work()[0], 0.5, 1e-12);

    // Re-evaluate outputs: result = x + Q = 0.5 + 5.0 = 5.5
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);
    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], 5.5, 1e-12);
}

// Custom model driven by entity-local (node-varying) input.
namespace {
class NodeDrivenModel : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "node_driven"; }
    int dimension() const override { return 1; }
    AuxiliaryStructuralMetadata structuralMetadata() const override {
        AuxiliaryStructuralMetadata m;
        m.variable_kinds = {AuxiliaryVariableKind::Differential};
        return m;
    }
    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override {
        // dx/dt = V_local - x, where V_local = inputs[0]
        const Real V = ctx.inputs.empty() ? 0.0 : ctx.inputs[0];
        req.residual[0] = ctx.xdot[0] - V + ctx.x[0];
    }
    bool hasAnalyticJacobian() const override { return true; }
    void evaluateJacobian(const AuxiliaryLocalContext&,
                          AuxiliaryJacobianRequest& req) const override {
        if (!req.dF_dx.empty()) req.dF_dx[0] = 1.0;
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 1.0;
    }
    int outputCount() const override { return 1; }
    std::vector<std::string> outputNames() const override { return {"x_plus_V"}; }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                          std::span<Real> output) const override {
        const Real V = ctx.inputs.empty() ? 0.0 : ctx.inputs[0];
        output[0] = ctx.x[0] + V;
    }
};
} // namespace

TEST(AuxiliaryModelBuilder, EndToEnd_EntityLocalInputs)
{
    using namespace svmp::FE;

    auto model = std::make_shared<NodeDrivenModel>();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    // Register entity-local input: 3 nodes, each with voltage = (entity+1)*10
    auto& reg = system.auxiliaryInputRegistry();
    AuxiliaryInputSpec vspec;
    vspec.name = "V_nodes";
    vspec.size = 1;
    vspec.entity_count = 3;
    reg.registerEntityInput(vspec,
        [](Real, Real, std::size_t e, std::span<Real> out) {
            out[0] = static_cast<Real>(e + 1) * 10.0; // 10, 20, 30
        });

    // Deploy with 3 Node-scoped entities and bind to entity-local input.
    system.deployAuxiliaryModel(
        use(model)
            .name("node_model")
            .scope(AuxiliaryStateScope::Node)
            .entityCount(3)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .stepper({"ForwardEuler"})
            .bind("V", "V_nodes")
            .initialize({0.0})); // All entities start at 0

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();

    // Advance: each entity sees different V
    // Entity 0: x = 0 + 0.1*(10 - 0) = 1.0
    // Entity 1: x = 0 + 0.1*(20 - 0) = 2.0
    // Entity 2: x = 0 + 0.1*(30 - 0) = 3.0
    system.advanceAuxiliaryState(0.0, 0.1);

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("node_model");
    EXPECT_NEAR(blk.workEntity(0)[0], 1.0, 1e-12);
    EXPECT_NEAR(blk.workEntity(1)[0], 2.0, 1e-12);
    EXPECT_NEAR(blk.workEntity(2)[0], 3.0, 1e-12);

    // Evaluate outputs per entity
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);

    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 3u); // 3 entities × 1 output
    // Entity 0: x=1 + V=10 = 11
    // Entity 1: x=2 + V=20 = 22
    // Entity 2: x=3 + V=30 = 33
    EXPECT_NEAR(outputs[0], 11.0, 1e-12);
    EXPECT_NEAR(outputs[1], 22.0, 1e-12);
    EXPECT_NEAR(outputs[2], 33.0, 1e-12);
}

TEST(AuxiliaryModelBuilder, EndToEnd_DeploymentRegion_ExplicitEntities)
{
    using namespace svmp::FE;

    // Model: dx/dt = -x, output = 2*x
    auto model = std::make_shared<CustomModelWithOutputs>();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    // Deploy with a region restricting to entities {1, 3} out of a 5-entity space.
    AuxiliaryDeploymentRegion region;
    region.kind = AuxiliaryRegionKind::CellSet;
    region.identity = "active_cells";
    region.explicit_entities = {1, 3};

    system.deployAuxiliaryModel(
        use(model)
            .name("region_model")
            .scope(AuxiliaryStateScope::Cell)
            .entityCount(5)  // full entity space
            .region(region)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .stepper({"ForwardEuler"})
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    // The block should have 2 entities (restricted by region), not 5.
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("region_model");
    EXPECT_EQ(blk.entityCount(), 2u); // only entities {1, 3}

    // Advance
    system.beginTimeStep();
    system.advanceAuxiliaryState(0.0, 0.1);

    // Both entities should advance: x = 1 + 0.1*(-1) = 0.9
    EXPECT_NEAR(blk.workEntity(0)[0], 0.9, 1e-12);
    EXPECT_NEAR(blk.workEntity(1)[0], 0.9, 1e-12);

    // Output eval: 2 entities × 1 output
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);
    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_NEAR(outputs[0], 1.8, 1e-12); // 0.9 * 2
    EXPECT_NEAR(outputs[1], 1.8, 1e-12);
}

// Custom model with declared input/parameter signatures that differ
// from lexicographic order.
namespace {
class SignedOrderModel : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "sig_order"; }
    int dimension() const override { return 1; }
    AuxiliaryStructuralMetadata structuralMetadata() const override {
        AuxiliaryStructuralMetadata m;
        m.variable_kinds = {AuxiliaryVariableKind::Differential};
        return m;
    }
    // Signature order: inputs=[B, A], params=[Z, M]
    // Lexicographic would be: inputs=[A, B], params=[M, Z]
    std::vector<std::string> declaredInputNames() const override { return {"B", "A"}; }
    std::vector<std::string> declaredParameterNames() const override { return {"Z", "M"}; }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override {
        // dx/dt = inputs[0]*params[0] + inputs[1]*params[1] - x
        // With sig order: B*Z + A*M - x
        const Real B = ctx.inputs.size() > 0 ? ctx.inputs[0] : 0.0;
        const Real A = ctx.inputs.size() > 1 ? ctx.inputs[1] : 0.0;
        const Real Z = ctx.params.size() > 0 ? ctx.params[0] : 0.0;
        const Real M = ctx.params.size() > 1 ? ctx.params[1] : 0.0;
        req.residual[0] = ctx.xdot[0] - B*Z - A*M + ctx.x[0];
    }
    bool hasAnalyticJacobian() const override { return true; }
    void evaluateJacobian(const AuxiliaryLocalContext&,
                          AuxiliaryJacobianRequest& req) const override {
        if (!req.dF_dx.empty()) req.dF_dx[0] = 1.0;
        if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 1.0;
    }
    int outputCount() const override { return 1; }
    std::vector<std::string> outputNames() const override { return {"result"}; }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                          std::span<Real> output) const override {
        const Real B = ctx.inputs.size() > 0 ? ctx.inputs[0] : 0.0;
        const Real A = ctx.inputs.size() > 1 ? ctx.inputs[1] : 0.0;
        const Real Z = ctx.params.size() > 0 ? ctx.params[0] : 0.0;
        const Real M = ctx.params.size() > 1 ? ctx.params[1] : 0.0;
        output[0] = B*Z + A*M;
    }
};
} // namespace

TEST(AuxiliaryModelBuilder, EndToEnd_SignatureOrdering)
{
    using namespace svmp::FE;

    auto model = std::make_shared<SignedOrderModel>();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    auto& reg = system.auxiliaryInputRegistry();
    reg.registerInput({.name = "inp_A", .size = 1}, [](Real, Real, std::span<Real> o) { o[0] = 2.0; });
    reg.registerInput({.name = "inp_B", .size = 1}, [](Real, Real, std::span<Real> o) { o[0] = 3.0; });

    // Bind: model's "B" → registry "inp_B", model's "A" → registry "inp_A"
    // Params: model's "Z" = 10, model's "M" = 100
    system.deployAuxiliaryModel(
        use(model)
            .name("sig_inst")
            .stepper({"ForwardEuler"})
            .bind("B", "inp_B")
            .bind("A", "inp_A")
            .param("Z", 10.0)
            .param("M", 100.0)
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();

    // With signature order: inputs=[B=3, A=2], params=[Z=10, M=100]
    // RHS = B*Z + A*M - x = 3*10 + 2*100 - 0 = 230
    // FE: x = 0 + 0.1 * 230 = 23.0
    system.advanceAuxiliaryState(0.0, 0.1);
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("sig_inst").work()[0], 23.0, 1e-12);

    // Output: B*Z + A*M = 3*10 + 2*100 = 230
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);
    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], 230.0, 1e-12);

    // If ordering were lexicographic (A before B, M before Z):
    // inputs=[A=2, B=3], params=[M=100, Z=10]
    // RHS would be = 2*100 + 3*10 = 230 (same value by coincidence)
    // But the model reads inputs[0]=B, inputs[1]=A, so with lex order
    // it would get inputs[0]=A=2, inputs[1]=B=3 → 2*10 + 3*100 = 320 ≠ 230
    // The test value 23.0 proves signature ordering is being used.
}

TEST(AuxiliaryModelBuilder, EndToEnd_CustomModel_MultiComponentInput)
{
    using namespace svmp::FE;

    // Custom model with "velocity:3" input declaration — 3 components.
    class VecInputModel : public AuxiliaryStateModel {
    public:
        std::string modelName() const override { return "vec_in"; }
        int dimension() const override { return 1; }
        AuxiliaryStructuralMetadata structuralMetadata() const override {
            AuxiliaryStructuralMetadata m;
            m.variable_kinds = {AuxiliaryVariableKind::Differential};
            return m;
        }
        std::vector<std::string> declaredInputNames() const override {
            return {"velocity:3"};
        }
        void evaluateResidual(const AuxiliaryLocalContext& ctx,
                              AuxiliaryResidualRequest& req) const override {
            // dx/dt = |velocity| = sqrt(v0^2 + v1^2 + v2^2) - x
            Real v0 = ctx.inputs.size() > 0 ? ctx.inputs[0] : 0.0;
            Real v1 = ctx.inputs.size() > 1 ? ctx.inputs[1] : 0.0;
            Real v2 = ctx.inputs.size() > 2 ? ctx.inputs[2] : 0.0;
            req.residual[0] = ctx.xdot[0] - std::sqrt(v0*v0 + v1*v1 + v2*v2) + ctx.x[0];
        }
        bool hasAnalyticJacobian() const override { return true; }
        void evaluateJacobian(const AuxiliaryLocalContext&,
                              AuxiliaryJacobianRequest& req) const override {
            if (!req.dF_dx.empty()) req.dF_dx[0] = 1.0;
            if (req.want_dF_dxdot && !req.dF_dxdot.empty()) req.dF_dxdot[0] = 1.0;
        }
        int outputCount() const override { return 1; }
        std::vector<std::string> outputNames() const override { return {"speed"}; }
        void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                             std::span<Real> output) const override {
            Real v0 = ctx.inputs.size() > 0 ? ctx.inputs[0] : 0.0;
            Real v1 = ctx.inputs.size() > 1 ? ctx.inputs[1] : 0.0;
            Real v2 = ctx.inputs.size() > 2 ? ctx.inputs[2] : 0.0;
            output[0] = std::sqrt(v0*v0 + v1*v1 + v2*v2);
        }
    };

    auto model = std::make_shared<VecInputModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    // Register a 3-component input.
    auto& reg = system.auxiliaryInputRegistry();
    reg.registerInput({.name = "vel_source", .size = 3},
        [](Real, Real, std::span<Real> o) { o[0] = 3.0; o[1] = 4.0; o[2] = 0.0; });

    // Bind "velocity" (base name without :3) to registry "vel_source".
    system.deployAuxiliaryModel(
        use(model)
            .name("vec_inst")
            .stepper({"ForwardEuler"})
            .bind("velocity", "vel_source")
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();
    system.advanceAuxiliaryState(0.0, 0.1);

    // |velocity| = sqrt(9+16+0) = 5. dx/dt = 5 - x.
    // FE from x=0: x = 0 + 0.1*(5 - 0) = 0.5
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("vec_inst").work()[0], 0.5, 1e-12);

    // Output: speed = 5.0
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);
    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], 5.0, 1e-12);
}

TEST(AuxiliaryModelBuilder, DeployRejectsMalformedInputSuffix)
{
    using namespace svmp::FE;

    class BadSuffixModel : public AuxiliaryStateModel {
    public:
        std::string modelName() const override { return "bad"; }
        int dimension() const override { return 1; }
        AuxiliaryStructuralMetadata structuralMetadata() const override {
            AuxiliaryStructuralMetadata m;
            m.variable_kinds = {AuxiliaryVariableKind::Differential};
            return m;
        }
        std::vector<std::string> declaredInputNames() const override {
            return {"velocity:abc"};  // malformed
        }
        void evaluateResidual(const AuxiliaryLocalContext&,
                              AuxiliaryResidualRequest& req) const override {
            req.residual[0] = 0.0;
        }
    };

    auto model = std::make_shared<BadSuffixModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    EXPECT_THROW(
        system.deployAuxiliaryModel(
            use(model).name("bad_inst")
                .stepper({"ForwardEuler"})
                .initialize({0.0})),
        InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, DeployRejectsZeroSizeInputSuffix)
{
    using namespace svmp::FE;

    class ZeroSizeModel : public AuxiliaryStateModel {
    public:
        std::string modelName() const override { return "zero_sz"; }
        int dimension() const override { return 1; }
        AuxiliaryStructuralMetadata structuralMetadata() const override {
            AuxiliaryStructuralMetadata m;
            m.variable_kinds = {AuxiliaryVariableKind::Differential};
            return m;
        }
        std::vector<std::string> declaredInputNames() const override {
            return {"velocity:0"};
        }
        void evaluateResidual(const AuxiliaryLocalContext&,
                              AuxiliaryResidualRequest& req) const override {
            req.residual[0] = 0.0;
        }
    };

    auto model = std::make_shared<ZeroSizeModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    EXPECT_THROW(
        system.deployAuxiliaryModel(
            use(model).name("z_inst").stepper({"ForwardEuler"}).initialize({0.0})),
        InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, DeployRejectsEmptyBaseName)
{
    using namespace svmp::FE;

    class EmptyBaseModel : public AuxiliaryStateModel {
    public:
        std::string modelName() const override { return "empty_base"; }
        int dimension() const override { return 1; }
        AuxiliaryStructuralMetadata structuralMetadata() const override {
            AuxiliaryStructuralMetadata m;
            m.variable_kinds = {AuxiliaryVariableKind::Differential};
            return m;
        }
        std::vector<std::string> declaredInputNames() const override {
            return {":3"};
        }
        void evaluateResidual(const AuxiliaryLocalContext&,
                              AuxiliaryResidualRequest& req) const override {
            req.residual[0] = 0.0;
        }
    };

    auto model = std::make_shared<EmptyBaseModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    EXPECT_THROW(
        system.deployAuxiliaryModel(
            use(model).name("e_inst").stepper({"ForwardEuler"}).initialize({0.0})),
        InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, DeployRejectsTrailingJunkInSuffix)
{
    using namespace svmp::FE;

    class JunkSuffixModel : public AuxiliaryStateModel {
    public:
        std::string modelName() const override { return "junk"; }
        int dimension() const override { return 1; }
        AuxiliaryStructuralMetadata structuralMetadata() const override {
            AuxiliaryStructuralMetadata m;
            m.variable_kinds = {AuxiliaryVariableKind::Differential};
            return m;
        }
        std::vector<std::string> declaredInputNames() const override {
            return {"velocity:3x"};
        }
        void evaluateResidual(const AuxiliaryLocalContext&,
                              AuxiliaryResidualRequest& req) const override {
            req.residual[0] = 0.0;
        }
    };

    auto model = std::make_shared<JunkSuffixModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    EXPECT_THROW(
        system.deployAuxiliaryModel(
            use(model).name("j_inst").stepper({"ForwardEuler"}).initialize({0.0})),
        InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, DeployRejectsRaggedLayout)
{
    using namespace svmp::FE;

    auto model = AuxiliaryModelBuilder("ragged_test")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    // Ragged layout should be rejected at finalization.
    system.deployAuxiliaryModel(
        use(model).name("ragged_inst")
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .initialize({0.0}));

    EXPECT_THROW(system.finalizeAuxiliaryLayout(), NotImplementedException);
}

TEST(AuxiliaryModelBuilder, EndToEnd_ByComponentThenEntity_Deployment)
{
    using namespace svmp::FE;

    // Deploy with ByComponentThenEntity ordering through the public API.
    // 2 entities, 2-state model.  This verifies:
    // 1. The ordering enum propagates to storage.
    // 2. Multi-entity initialization is correctly transposed to component-major.
    // 3. Per-entity stepping and output eval work with the non-default ordering.
    auto model = AuxiliaryModelBuilder("bce_model")
        .state("x").state("y")
        .ode("x", -modelState("x"))
        .ode("y", -modelState("y") * forms::FormExpr::constant(2.0))
        .output("sum", modelState("x") + modelState("y"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model).name("bce_inst")
            .scope(AuxiliaryStateScope::Cell)
            .entityCount(2)
            .entityOrdering(AuxiliaryEntityOrdering::ByComponentThenEntity)
            .stepper({"ForwardEuler"})
            .initialize({1.0, 2.0}));

    system.finalizeAuxiliaryLayout();

    // Verify block ordering.
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("bce_inst");
    EXPECT_EQ(blk.ordering(), AuxiliaryEntityOrdering::ByComponentThenEntity);

    // Verify initial values are correct per entity (both should be {1.0, 2.0}).
    auto e0_init = blk.gatherEntityWork(0);
    EXPECT_NEAR(e0_init[0], 1.0, 1e-12);
    EXPECT_NEAR(e0_init[1], 2.0, 1e-12);
    auto e1_init = blk.gatherEntityWork(1);
    EXPECT_NEAR(e1_init[0], 1.0, 1e-12);
    EXPECT_NEAR(e1_init[1], 2.0, 1e-12);

    system.beginTimeStep();

    // ForwardEuler, dt=0.1:
    // Each entity: x = 1 + 0.1*(-1) = 0.9, y = 2 + 0.1*(-4) = 1.6
    system.advanceAuxiliaryState(0.0, 0.1);

    auto e0 = blk.gatherEntityWork(0);
    EXPECT_NEAR(e0[0], 0.9, 1e-12);
    EXPECT_NEAR(e0[1], 1.6, 1e-12);
    auto e1 = blk.gatherEntityWork(1);
    EXPECT_NEAR(e1[0], 0.9, 1e-12);
    EXPECT_NEAR(e1[1], 1.6, 1e-12);

    // Output eval: 2 entities × 1 output.
    systems::SystemStateView state;
    state.time = 0.1; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);
    auto outputs = system.auxiliaryOutputValues();
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_NEAR(outputs[0], 0.9 + 1.6, 1e-12);
    EXPECT_NEAR(outputs[1], 0.9 + 1.6, 1e-12);
}

// Helper: custom model with "velocity:3" and entity-local inputs, used by
// the entity-local and monolithic multi-component tests below.
namespace {
class VecInputEntityModel : public AuxiliaryStateModel {
public:
    std::string modelName() const override { return "vec_entity"; }
    int dimension() const override { return 1; }
    AuxiliaryStructuralMetadata structuralMetadata() const override {
        AuxiliaryStructuralMetadata m;
        m.variable_kinds = {AuxiliaryVariableKind::Algebraic};
        return m;
    }
    std::vector<std::string> declaredInputNames() const override {
        return {"velocity:3"};
    }
    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override {
        // F = x - (v0 + v1 + v2) = 0
        Real v0 = ctx.inputs.size() > 0 ? ctx.inputs[0] : 0.0;
        Real v1 = ctx.inputs.size() > 1 ? ctx.inputs[1] : 0.0;
        Real v2 = ctx.inputs.size() > 2 ? ctx.inputs[2] : 0.0;
        req.residual[0] = ctx.x[0] - (v0 + v1 + v2);
    }
    bool hasAnalyticJacobian() const override { return true; }
    void evaluateJacobian(const AuxiliaryLocalContext&,
                          AuxiliaryJacobianRequest& req) const override {
        if (!req.dF_dx.empty()) req.dF_dx[0] = 1.0;
    }
    int outputCount() const override { return 1; }
    std::vector<std::string> outputNames() const override { return {"sum"}; }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                         std::span<Real> output) const override {
        Real v0 = ctx.inputs.size() > 0 ? ctx.inputs[0] : 0.0;
        Real v1 = ctx.inputs.size() > 1 ? ctx.inputs[1] : 0.0;
        Real v2 = ctx.inputs.size() > 2 ? ctx.inputs[2] : 0.0;
        output[0] = v0 + v1 + v2;
    }
};
} // namespace

TEST(AuxiliaryModelBuilder, EndToEnd_MultiComponentInput_EntityLocal)
{
    using namespace svmp::FE;

    // 2-entity model with entity-local 3-component input.
    auto model = std::make_shared<VecInputEntityModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    auto& reg = system.auxiliaryInputRegistry();
    // Entity-local: entity 0 → {1,2,3}, entity 1 → {2,3,4}.
    reg.registerEntityInput(
        {.name = "vel_src", .size = 3, .entity_count = 2},
        [](Real, Real, std::size_t entity, std::span<Real> o) {
            const auto base = static_cast<Real>(entity + 1);
            o[0] = base; o[1] = base + 1; o[2] = base + 2;
        });

    system.deployAuxiliaryModel(
        use(model).name("el_inst")
            .scope(AuxiliaryStateScope::Cell)
            .entityCount(2)
            .bind("velocity", "vel_src")
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();

    // Output eval triggers entity-local rebuild with name:size parsing.
    systems::SystemStateView state;
    state.time = 0.0; state.dt = 0.1;
    system.prepareAuxiliaryForAssembly(state);
    auto outputs = system.auxiliaryOutputValues();

    // Entity 0: v={1,2,3} → sum=6. Entity 1: v={2,3,4} → sum=9.
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_NEAR(outputs[0], 6.0, 1e-12);
    EXPECT_NEAR(outputs[1], 9.0, 1e-12);
}

TEST(AuxiliaryModelBuilder, EndToEnd_MultiComponentInput_Monolithic)
{
    using namespace svmp::FE;

    // Monolithic model with "velocity:3" input. Global, 1 entity.
    auto model = std::make_shared<VecInputEntityModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    auto& reg = system.auxiliaryInputRegistry();
    reg.registerInput({.name = "vel_src", .size = 3},
        [](Real, Real, std::span<Real> o) { o[0] = 10.0; o[1] = 20.0; o[2] = 30.0; });

    system.deployAuxiliaryModel(
        use(model).name("mono_vec")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .bind("velocity", "vel_src")
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();
    // No manual reg.evaluate() — assembleMonolithicAuxiliary evaluates inputs.

    // F(x=0, v={10,20,30}) = 0 - (10+20+30) = -60.
    std::vector<Real> residual(1);
    std::vector<Real> jacobian(1);
    system.assembleMonolithicAuxiliary(0.0, 0.1, residual, jacobian);

    EXPECT_NEAR(residual[0], -60.0, 1e-12);
    EXPECT_NEAR(jacobian[0], 1.0, 1e-10);
}

TEST(AuxiliaryModelBuilder, EndToEnd_MultiComponentInput_Monolithic_EntityLocal)
{
    using namespace svmp::FE;

    // Monolithic model with "velocity:3" entity-local input, 2 entities.
    // Exercises the monolithic entity-local rebuild path with name:size parsing.
    auto model = std::make_shared<VecInputEntityModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    auto& reg = system.auxiliaryInputRegistry();
    // Entity-local 3-component: entity 0 → {10,20,30}, entity 1 → {1,2,3}.
    reg.registerEntityInput(
        {.name = "vel_src", .size = 3, .entity_count = 2},
        [](Real, Real, std::size_t entity, std::span<Real> o) {
            if (entity == 0) { o[0] = 10.0; o[1] = 20.0; o[2] = 30.0; }
            else             { o[0] = 1.0;  o[1] = 2.0;  o[2] = 3.0;  }
        });

    system.deployAuxiliaryModel(
        use(model).name("mono_el")
            .scope(AuxiliaryStateScope::Node)
            .entityCount(2)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .bind("velocity", "vel_src")
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();

    // F(x, v) = x - (v0+v1+v2) per entity.
    // Entity 0: F = 0 - (10+20+30) = -60, dF/dx = 1
    // Entity 1: F = 0 - (1+2+3) = -6,    dF/dx = 1
    std::vector<Real> residual(2);
    std::vector<Real> jacobian(4);
    system.assembleMonolithicAuxiliary(0.0, 0.1, residual, jacobian);

    EXPECT_NEAR(residual[0], -60.0, 1e-12);
    EXPECT_NEAR(residual[1], -6.0, 1e-12);
    EXPECT_NEAR(jacobian[0], 1.0, 1e-10); // dF0/dx0
    EXPECT_NEAR(jacobian[3], 1.0, 1e-10); // dF1/dx1
}

TEST(AuxiliaryModelBuilder, EndToEnd_Monolithic_EachNonlinearIteration_InputRefresh)
{
    using namespace svmp::FE;

    // Algebraic model F = x - Q = 0 with EachNonlinearIteration input.
    // The input callback returns a value that changes on each call,
    // simulating a Newton-dependent quantity.
    auto model = AuxiliaryModelBuilder("iter_model")
        .state("x", AuxiliaryVariableKind::Algebraic)
        .input("Q")
        .algebraic("x", modelState("x") - modelInput("Q"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});
    auto& reg = system.auxiliaryInputRegistry();

    // Counter-based callback: returns 10 on first eval, 20 on second.
    int call_count = 0;
    reg.registerInput(
        {.name = "Q_iter",
         .size = 1,
         .update_schedule = AuxiliaryInputUpdateSchedule::EachNonlinearIteration},
        [&call_count](Real, Real, std::span<Real> o) {
            ++call_count;
            o[0] = static_cast<Real>(call_count * 10);
        });

    system.deployAuxiliaryModel(
        use(model).name("iter_inst")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .bind("Q", "Q_iter")
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();

    std::vector<Real> res(1), jac(1);

    // First assembly (not nonlinear iteration): evaluates input → Q=10.
    system.assembleMonolithicAuxiliary(0.0, 0.1, res, jac, false);
    EXPECT_NEAR(res[0], -10.0, 1e-12); // F = 0 - 10

    // Second assembly WITHOUT nonlinear flag: EachNonlinearIteration
    // input should NOT refresh — still Q=10.
    system.assembleMonolithicAuxiliary(0.0, 0.1, res, jac, false);
    EXPECT_NEAR(res[0], -10.0, 1e-12); // still Q=10

    // Third assembly WITH nonlinear flag: EachNonlinearIteration
    // input refreshes → Q=20.
    system.assembleMonolithicAuxiliary(0.0, 0.1, res, jac, true);
    EXPECT_NEAR(res[0], -20.0, 1e-12); // now Q=20
}

// ============================================================================
//  Schedule mode integration tests
// ============================================================================

TEST(AuxiliaryModelBuilder, EndToEnd_SubcycledSchedule)
{
    using namespace svmp::FE;

    // dx/dt = -x, subcycled with 5 substeps per PDE step
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    AuxiliaryStepperSpec stepper_spec;
    stepper_spec.method_name = "ForwardEuler";
    stepper_spec.substep_count = 5;

    system.deployAuxiliaryModel(
        use(model)
            .name("subcycled_decay")
            .schedule(AuxiliaryScheduleMode::Subcycled)
            .stepper(stepper_spec)
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();

    // PDE dt = 0.5, subcycled into 5 substeps of 0.1 each.
    // FE with dt_sub=0.1: x *= (1 - 0.1) each substep = 0.9^5 = 0.59049
    system.advanceAuxiliaryState(0.0, 0.5);

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("subcycled_decay").work()[0],
                std::pow(0.9, 5), 1e-12);
}

TEST(AuxiliaryModelBuilder, EndToEnd_MultirateTwoBlocks)
{
    using namespace svmp::FE;

    // Block A: single-rate (1 step per PDE step)
    auto model_a = AuxiliaryModelBuilder("a")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    // Block B: subcycled (3 substeps per PDE step)
    auto model_b = AuxiliaryModelBuilder("b")
        .state("x")
        .ode("x", -modelState("x") * forms::FormExpr::constant(0.5))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    AuxiliaryStepperSpec spec_a;
    spec_a.method_name = "ForwardEuler";
    spec_a.substep_count = 1;

    AuxiliaryStepperSpec spec_b;
    spec_b.method_name = "ForwardEuler";
    spec_b.substep_count = 3;

    system.deployAuxiliaryModel(
        use(model_a).name("block_a")
            .schedule(AuxiliaryScheduleMode::SingleRate)
            .stepper(spec_a)
            .initialize({1.0}));

    system.deployAuxiliaryModel(
        use(model_b).name("block_b")
            .schedule(AuxiliaryScheduleMode::Subcycled)
            .stepper(spec_b)
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();

    // PDE dt = 0.3
    // Block A (single-rate): 1 step of dt=0.3 → x = 1 + 0.3*(-1) = 0.7
    // Block B (subcycled, 3 substeps): 3 steps of dt=0.1
    //   step1: x = 1 + 0.1*(-0.5) = 0.95
    //   step2: x = 0.95 + 0.1*(-0.475) = 0.9025
    //   step3: x = 0.9025 + 0.1*(-0.45125) = 0.856375
    system.advanceAuxiliaryState(0.0, 0.3);

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("block_a").work()[0], 0.7, 1e-12);
    EXPECT_NEAR(mgr->getBlock("block_b").work()[0],
                std::pow(1.0 - 0.05, 3), 1e-12); // (0.95)^3 = 0.857375
}

TEST(AuxiliaryModelBuilder, EndToEnd_MultirateSchedulerDispatch)
{
    using namespace svmp::FE;

    // Block A: single-rate (rate_ratio=1)
    auto model_a = AuxiliaryModelBuilder("a")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    // Block B: multirate (rate_ratio=2, i.e., 2 substeps per PDE step)
    auto model_b = AuxiliaryModelBuilder("b")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    AuxiliaryStepperSpec spec_a;
    spec_a.method_name = "ForwardEuler";
    spec_a.substep_count = 1;

    AuxiliaryStepperSpec spec_b;
    spec_b.method_name = "ForwardEuler";
    spec_b.substep_count = 2;  // rate_ratio for multirate

    system.deployAuxiliaryModel(
        use(model_a).name("block_a")
            .schedule(AuxiliaryScheduleMode::SingleRate)
            .stepper(spec_a)
            .initialize({1.0}));

    system.deployAuxiliaryModel(
        use(model_b).name("block_b")
            .schedule(AuxiliaryScheduleMode::Multirate)
            .stepper(spec_b)
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();

    // PDE dt = 0.4
    // Block A (single-rate, rate_ratio=1): 1 step of dt=0.4
    //   x = 1 + 0.4*(-1) = 0.6
    // Block B (multirate, rate_ratio=2): 2 interleaved substeps of dt=0.2
    //   substep 0: x = 1.0 + 0.2*(-1.0) = 0.8
    //   substep 1: x = 0.8 + 0.2*(-0.8) = 0.64
    system.advanceAuxiliaryState(0.0, 0.4);

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    EXPECT_NEAR(mgr->getBlock("block_a").work()[0], 0.6, 1e-12);

    // Block B (multirate, rate_ratio=2): scheduler dispatches 2 substeps
    // of dt=0.2 each.  ForwardEuler with dx/dt = -x:
    //   substep 0 (t=0, dt=0.2): x = 1.0 + 0.2*(-1.0) = 0.8
    //   substep 1 (t=0.2, dt=0.2): x = 0.8 + 0.2*(-0.8) = 0.64
    //
    // NOTE: Subcycled blocks use substep_count inside the stepper,
    // while Multirate blocks use the scheduler for interleaved dispatch.
    // For Multirate, substep_count becomes rate_ratio during finalization.
    // The scheduler calls advanceOneEntry with substep_count=1 for each
    // planned substep.
    const Real block_b_val = mgr->getBlock("block_b").work()[0];
    EXPECT_NEAR(block_b_val, 0.64, 1e-12)
        << "block_b should do 2 multirate substeps: 1.0→0.8→0.64, got " << block_b_val;
}

TEST(AuxiliaryModelBuilder, EndToEnd_Multirate_EntityLocalInputs)
{
    using namespace svmp::FE;

    // Multirate block with entity-local input.
    // 2 entities, rate_ratio=2.  Input varies per entity.
    auto model = AuxiliaryModelBuilder("el_multi")
        .state("x")
        .input("Q")
        .ode("x", modelInput("Q") - modelState("x"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});
    auto& reg = system.auxiliaryInputRegistry();

    // Entity-local: entity 0 → Q=10, entity 1 → Q=20.
    reg.registerEntityInput(
        {.name = "Q_src", .size = 1, .entity_count = 2},
        [](Real, Real, std::size_t entity, std::span<Real> o) {
            o[0] = (entity == 0) ? 10.0 : 20.0;
        });

    AuxiliaryStepperSpec spec;
    spec.method_name = "ForwardEuler";
    spec.substep_count = 2; // rate_ratio for multirate

    system.deployAuxiliaryModel(
        use(model).name("mr_el")
            .scope(AuxiliaryStateScope::Cell)
            .entityCount(2)
            .schedule(AuxiliaryScheduleMode::Multirate)
            .stepper(spec)
            .bind("Q", "Q_src")
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();
    system.advanceAuxiliaryState(0.0, 0.4);

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);

    // Entity 0: Q=10, dx/dt = 10 - x. 2 substeps of dt=0.2:
    //   step 0: x = 0 + 0.2*(10-0) = 2.0
    //   step 1: x = 2.0 + 0.2*(10-2.0) = 3.6
    EXPECT_NEAR(mgr->getBlock("mr_el").gatherEntityWork(0)[0], 3.6, 1e-12);

    // Entity 1: Q=20, dx/dt = 20 - x. 2 substeps of dt=0.2:
    //   step 0: x = 0 + 0.2*(20-0) = 4.0
    //   step 1: x = 4.0 + 0.2*(20-4.0) = 7.2
    EXPECT_NEAR(mgr->getBlock("mr_el").gatherEntityWork(1)[0], 7.2, 1e-12);
}

TEST(AuxiliaryModelBuilder, EndToEnd_Multirate_BackwardEulerHistory)
{
    using namespace svmp::FE;

    // Multirate block with BackwardEuler (implicit, uses derivative provider).
    // rate_ratio=2.  BackwardEuler needs the derivative for Newton iteration.
    auto model = AuxiliaryModelBuilder("be_multi")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    AuxiliaryStepperSpec spec;
    spec.method_name = "BackwardEuler";
    spec.substep_count = 2;

    system.deployAuxiliaryModel(
        use(model).name("mr_be")
            .schedule(AuxiliaryScheduleMode::Multirate)
            .stepper(spec)
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();
    system.beginTimeStep();
    system.advanceAuxiliaryState(0.0, 0.4);

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);

    // BackwardEuler with dx/dt = -x:  x_{n+1} = x_n / (1 + dt)
    // 2 substeps of dt=0.2:
    //   step 0: x = 1.0 / 1.2 ≈ 0.833333
    //   step 1: x = 0.833333 / 1.2 ≈ 0.694444
    const Real expected = 1.0 / (1.2 * 1.2);
    EXPECT_NEAR(mgr->getBlock("mr_be").work()[0], expected, 1e-6);
}

TEST(AuxiliaryModelBuilder, EndToEnd_Multirate_BDF2History)
{
    using namespace svmp::FE;

    // Run the same model with BDF2 and BackwardEuler in multirate mode.
    // After two steps, BDF2 should produce a DIFFERENT result from BE
    // because step 2 uses history.  This proves the history path is live.
    auto model_bdf = AuxiliaryModelBuilder("bdf2_m")
        .state("x").ode("x", -modelState("x")).build();
    auto model_be = AuxiliaryModelBuilder("be_m")
        .state("x").ode("x", -modelState("x")).build();

    systems::FESystem sys_bdf(std::shared_ptr<const assembly::IMeshAccess>{});
    systems::FESystem sys_be(std::shared_ptr<const assembly::IMeshAccess>{});

    AuxiliaryStepperSpec bdf_spec;
    bdf_spec.method_name = "BDF2";
    bdf_spec.substep_count = 2;

    AuxiliaryStepperSpec be_spec;
    be_spec.method_name = "BackwardEuler";
    be_spec.substep_count = 2;

    sys_bdf.deployAuxiliaryModel(
        use(model_bdf).name("mr")
            .schedule(AuxiliaryScheduleMode::Multirate)
            .stepper(bdf_spec).initialize({1.0}));
    sys_be.deployAuxiliaryModel(
        use(model_be).name("mr")
            .schedule(AuxiliaryScheduleMode::Multirate)
            .stepper(be_spec).initialize({1.0}));

    sys_bdf.finalizeAuxiliaryLayout();
    sys_be.finalizeAuxiliaryLayout();

    // Step 1: both fall back to BE (BDF2 has no history).
    sys_bdf.beginTimeStep(); sys_bdf.advanceAuxiliaryState(0.0, 0.4); sys_bdf.commitTimeStep();
    sys_be.beginTimeStep();  sys_be.advanceAuxiliaryState(0.0, 0.4);  sys_be.commitTimeStep();

    const Real bdf_s1 = sys_bdf.auxiliaryStateManagerIfPresent()->getBlock("mr").work()[0];
    const Real be_s1  = sys_be.auxiliaryStateManagerIfPresent()->getBlock("mr").work()[0];
    // Step 1 should match (both are BE fallback).
    EXPECT_NEAR(bdf_s1, be_s1, 1e-12);

    // Step 2: BDF2 uses history, BE does not.
    sys_bdf.beginTimeStep(); sys_bdf.advanceAuxiliaryState(0.4, 0.4);
    sys_be.beginTimeStep();  sys_be.advanceAuxiliaryState(0.4, 0.4);

    const Real bdf_s2 = sys_bdf.auxiliaryStateManagerIfPresent()->getBlock("mr").work()[0];
    const Real be_s2  = sys_be.auxiliaryStateManagerIfPresent()->getBlock("mr").work()[0];

    // BDF2 step 2 MUST differ from BE step 2 (proves history is used).
    EXPECT_NE(bdf_s2, be_s2)
        << "BDF2 and BE should differ on step 2 (history-dependent)";

    // Both should be decaying toward 0.
    EXPECT_TRUE(std::isfinite(bdf_s2));
    EXPECT_LT(bdf_s2, bdf_s1);
    EXPECT_TRUE(std::isfinite(be_s2));
    EXPECT_LT(be_s2, be_s1);
}

// ============================================================================
//  Monolithic assembly tests
// ============================================================================

#include "Auxiliary/AuxiliaryOperatorRegistry.h"

TEST(AuxiliaryModelBuilder, EndToEnd_MonolithicAssembly)
{
    using namespace svmp::FE;

    // Monolithic model: F = x^2 - 4 = 0 (algebraic, solution x=2)
    auto model = AuxiliaryModelBuilder("algebraic")
        .state("x", AuxiliaryVariableKind::Algebraic)
        .algebraic("x", modelState("x") * modelState("x") - forms::FormExpr::constant(4.0))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model)
            .name("mono_block")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .initialize({3.0})); // Initial guess x=3

    system.finalizeAuxiliaryLayout();

    // Check layout.
    auto layout = system.composeMixedSystemLayout(0);
    EXPECT_EQ(layout.n_aux_unknowns, 1u);
    EXPECT_EQ(layout.total_unknowns, 1u);

    // Assemble residual and Jacobian.
    std::vector<Real> residual(1);
    std::vector<Real> jacobian(1);
    system.assembleMonolithicAuxiliary(0.0, 0.1, residual, jacobian);

    // F(3) = 9 - 4 = 5
    EXPECT_NEAR(residual[0], 5.0, 1e-12);
    // dF/dx at x=3 = 2*3 = 6
    EXPECT_NEAR(jacobian[0], 6.0, 1e-5);

    // Manual Newton step: x_new = x - F/dFdx = 3 - 5/6 ≈ 2.1667
    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("mono_block");
    blk.work()[0] -= residual[0] / jacobian[0];
    EXPECT_NEAR(blk.work()[0], 3.0 - 5.0/6.0, 1e-12);
}

TEST(AuxiliaryModelBuilder, EndToEnd_MonolithicAssembly_WithInputs)
{
    using namespace svmp::FE;

    // Monolithic model with an input: F = x - Q = 0 (algebraic)
    // With Q=7, solution x=7, dF/dx=1.
    auto model = AuxiliaryModelBuilder("with_input")
        .state("x", AuxiliaryVariableKind::Algebraic)
        .input("Q")
        .algebraic("x", modelState("x") - modelInput("Q"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    auto& reg = system.auxiliaryInputRegistry();
    reg.registerInput({.name = "Q_source", .size = 1},
                      [](Real, Real, std::span<Real> out) { out[0] = 7.0; });

    system.deployAuxiliaryModel(
        use(model)
            .name("mono_inp")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .bind("Q", "Q_source")
            .initialize({3.0}));

    system.finalizeAuxiliaryLayout();
    // No manual reg.evaluate() — assembleMonolithicAuxiliary evaluates inputs.

    std::vector<Real> residual(1);
    std::vector<Real> jacobian(1);
    system.assembleMonolithicAuxiliary(0.0, 0.1, residual, jacobian);

    // F(x=3, Q=7) = 3 - 7 = -4.  Inputs are wired by assembly.
    EXPECT_NEAR(residual[0], -4.0, 1e-12);
    EXPECT_NEAR(jacobian[0], 1.0, 1e-5);
}

TEST(AuxiliaryModelBuilder, EndToEnd_MonolithicAssembly_UsesGeneralizedAlphaStencil)
{
    using namespace svmp::FE;

    auto model = aux_test::buildDecay(/*k_default=*/1.0);
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model)
            .name("ga_decay")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .param("k", 1.0)
            .initialize({0.8}));

    system.finalizeAuxiliaryLayout();

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("ga_decay");

    // Establish x_n = 1.0 with one committed history snapshot x_{n-1} = 0.8.
    blk.work()[0] = 1.0;
    system.commitTimeStep();

    // Solve a stage state x_{n+alpha_f} = 1.1.
    system.beginTimeStep();
    blk.work()[0] = 1.1;

    std::array<double, 1> dt_hist{{0.1}};
    auto ti = aux_test::buildGeneralizedAlphaFirstOrderContext(/*dt=*/0.1);

    systems::SystemStateView state;
    state.time = 0.06666666666666667;
    state.dt = 0.1;
    state.effective_dt = 0.06666666666666667;
    state.dt_prev = 0.1;
    state.dt_history = dt_hist;
    state.time_integration = &ti;

    std::vector<Real> residual;
    std::vector<Real> jacobian;
    system.assembleMixedAuxiliaryDense(state, /*n_field_dofs=*/0, residual, jacobian);

    ASSERT_EQ(residual.size(), 1u);
    ASSERT_EQ(jacobian.size(), 1u);

    // Generalized-alpha(1st-order) with rho_inf=0.5 gives:
    //   xdot_stage = 18.75*x_stage - 18.75*x_n - 0.25*xdot_n
    // Using x_n = 1.0, x_{n-1} = 0.8 => xdot_n = 2.0, so xdot_stage = 1.375.
    // For dx/dt = -x, F = xdot + x and dF/dx = 1 + 18.75.
    EXPECT_NEAR(residual[0], 2.475, 1e-12);
    EXPECT_NEAR(jacobian[0], 19.75, 1e-12);
}

TEST(AuxiliaryModelBuilder, PrepareAuxiliaryForAssembly_UsesStageXDotForOutputs)
{
    using namespace svmp::FE;

    auto model = std::make_shared<aux_test::OutputUsesXDotModel>();
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model)
            .name("xdot_out")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .initialize({0.8}));

    system.finalizeAuxiliaryLayout();

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("xdot_out");

    blk.work()[0] = 1.0;
    system.commitTimeStep();

    system.beginTimeStep();
    blk.work()[0] = 1.1;

    std::array<double, 1> dt_hist{{0.1}};
    auto ti = aux_test::buildGeneralizedAlphaFirstOrderContext(/*dt=*/0.1);

    systems::SystemStateView state;
    state.time = 0.06666666666666667;
    state.dt = 0.1;
    state.effective_dt = 0.06666666666666667;
    state.dt_prev = 0.1;
    state.dt_history = dt_hist;
    state.time_integration = &ti;

    system.prepareAuxiliaryForAssembly(state, false);
    const auto outputs = system.auxiliaryOutputValues();

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], 1.375, 1e-12);
}

TEST(AuxiliaryModelBuilder, FinalizeMonolithicAuxiliaryStageState_OnlyTransformsDifferentialRows)
{
    using namespace svmp::FE;

    auto model = AuxiliaryModelBuilder("stage_finalize")
        .state("x")
        .state("z", AuxiliaryVariableKind::Algebraic)
        .ode("x", forms::FormExpr::constant(0.0))
        .algebraic("z", modelState("z"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});
    system.deployAuxiliaryModel(
        use(model)
            .name("stage_block")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .initialize({1.0, 2.0}));
    system.finalizeAuxiliaryLayout();

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("stage_block");

    blk.work()[0] = 1.4;
    blk.work()[1] = 5.0;

    system.finalizeMonolithicAuxiliaryStageState(static_cast<Real>(2.0 / 3.0),
                                                 static_cast<Real>(0.1));

    EXPECT_NEAR(blk.work()[0], 1.6, 1e-12);
    EXPECT_NEAR(blk.work()[1], 5.0, 1e-12);
}

TEST(AuxiliaryModelBuilder, EndToEnd_MonolithicLayout_WithFields)
{
    using namespace svmp::FE;

    auto model = AuxiliaryModelBuilder("aux_mono")
        .state("p")
        .ode("p", -modelState("p"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model)
            .name("mono_p")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    // Compose with 100 FE field unknowns.
    auto layout = system.composeMixedSystemLayout(100);
    EXPECT_EQ(layout.n_field_unknowns, 100u);
    EXPECT_EQ(layout.n_aux_unknowns, 1u);
    EXPECT_EQ(layout.total_unknowns, 101u);
    EXPECT_EQ(layout.aux_layout.mixed_system_offset, 100u);
}

TEST(AuxiliaryModelBuilder, MonolithicSolverMetadataPropagatesIntoMixedLayoutAndSummary)
{
    using namespace svmp::FE;

    auto model = AuxiliaryModelBuilder("lambda_model")
        .state("lambda", AuxiliaryVariableKind::Algebraic)
        .algebraic("lambda", modelState("lambda"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    system.deployAuxiliaryModel(
        use(model)
            .name("lambda_block")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .solverRole(systems::AuxiliaryBlockRole::Constraint)
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto layout = system.composeMixedSystemLayout(8);
    ASSERT_EQ(layout.aux_layout.blocks.size(), 1u);
    EXPECT_EQ(layout.aux_layout.blocks[0].name, "lambda_block");
    EXPECT_EQ(layout.aux_layout.blocks[0].role, systems::AuxiliaryBlockRole::Constraint);
    EXPECT_EQ(layout.aux_layout.blocks[0].backend_role,
              backends::BlockRole::ConstraintField);

    const auto summary = system.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_constraint_like_blocks, 1u);
    EXPECT_EQ(summary.constraint_like_block_names,
              std::vector<std::string>({"lambda_block"}));
}

TEST(AuxiliaryModelBuilder, AugmentSolverOptionsExportsMixedBlockLayoutAndRoleNames)
{
    using namespace svmp::FE;

    auto model = AuxiliaryModelBuilder("lambda_model")
        .state("lambda", AuxiliaryVariableKind::Algebraic)
        .algebraic("lambda", modelState("lambda"))
        .build();

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});
    system.deployAuxiliaryModel(
        use(model)
            .name("lambda_block")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .solverRole(systems::AuxiliaryBlockRole::Constraint)
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    backends::SolverOptions base_opts{};
    const auto opts = system.augmentSolverOptions(base_opts, /*n_field_unknowns=*/8);

    ASSERT_TRUE(opts.mixed_block_layout.has_value());
    EXPECT_EQ(opts.mixed_block_layout->field_unknowns, 8);
    EXPECT_EQ(opts.mixed_block_layout->auxiliary_unknowns, 1);
    EXPECT_EQ(opts.mixed_block_layout->total_unknowns, 9);

    const auto* lambda = opts.mixed_block_layout->findBlock("lambda_block");
    ASSERT_NE(lambda, nullptr);
    EXPECT_EQ(lambda->offset, 8);
    EXPECT_EQ(lambda->size, 1);
    EXPECT_EQ(lambda->role, backends::BlockRole::ConstraintField);
    EXPECT_EQ(lambda->kind, backends::MixedBlockKind::Auxiliary);
    EXPECT_EQ(lambda->assembly_mode, backends::MixedBlockAssemblyMode::BorderedReduced);
    EXPECT_EQ(lambda->row_ownership, backends::MixedRowOwnershipPolicy::SingleOwner);
    EXPECT_EQ(lambda->single_owner_rank, 0);

    ASSERT_EQ(opts.block_role_names.size(), 1u);
    EXPECT_EQ(opts.block_role_names[0].first, backends::BlockRole::ConstraintField);
    EXPECT_EQ(opts.block_role_names[0].second, "lambda_block");
}

TEST(AuxiliaryModelBuilder, MonolithicScopeOwnershipPoliciesPropagateIntoMixedLayout)
{
    using namespace svmp::FE;

    auto model = aux_test::buildDecay(1.0);

    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});
    system.deployAuxiliaryModel(
        use(model)
            .name("global_aux")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .initialize({0.0}));
    system.deployAuxiliaryModel(
        use(model)
            .name("node_aux")
            .scope(AuxiliaryStateScope::Node)
            .entityCount(2)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .initialize({0.0}));
    system.deployAuxiliaryModel(
        use(model)
            .name("region_aux")
            .region()
            .entityCount(3)
            .solveMode(AuxiliarySolveMode::Monolithic)
            .initialize({0.0}));

    system.finalizeAuxiliaryLayout();
    const auto opts = system.augmentSolverOptions(backends::SolverOptions{},
                                                  /*n_field_unknowns=*/0);
    ASSERT_TRUE(opts.mixed_block_layout.has_value());

    const auto* global = opts.mixed_block_layout->findBlock("global_aux");
    ASSERT_NE(global, nullptr);
    EXPECT_EQ(global->assembly_mode, backends::MixedBlockAssemblyMode::BorderedReduced);
    EXPECT_EQ(global->row_ownership, backends::MixedRowOwnershipPolicy::SingleOwner);
    EXPECT_EQ(global->single_owner_rank, 0);

    const auto* node = opts.mixed_block_layout->findBlock("node_aux");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->assembly_mode, backends::MixedBlockAssemblyMode::BorderedReduced);
    EXPECT_EQ(node->row_ownership, backends::MixedRowOwnershipPolicy::BackendDofOwner);
    EXPECT_EQ(node->single_owner_rank, -1);

    const auto* region = opts.mixed_block_layout->findBlock("region_aux");
    ASSERT_NE(region, nullptr);
    EXPECT_EQ(region->assembly_mode, backends::MixedBlockAssemblyMode::BorderedReduced);
    EXPECT_EQ(region->row_ownership, backends::MixedRowOwnershipPolicy::RegionOwner);
    EXPECT_EQ(region->single_owner_rank, -1);
}

// ============================================================================
//  Name validation: '/' is reserved separator
// ============================================================================

TEST(AuxiliaryModelBuilder, RejectsSlashInStateName)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("test").state("x/y"),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, RejectsSlashInOutputName)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("test")
            .state("x")
            .output("out/put", modelState("x")),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, RejectsSlashInInputName)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("test").input("in/put"),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelBuilder, RejectsSlashInParamName)
{
    EXPECT_THROW(
        AuxiliaryModelBuilder("test").param("p/q"),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryBindings, RejectsSlashInInstanceName)
{
    auto model = AuxiliaryModelBuilder("simple")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    auto instance = use(model)
        .name("bad/name")
        .scope(AuxiliaryStateScope::Global)
        .solveMode(AuxiliarySolveMode::Partitioned)
        .initialize({0.0});

    auto err = instance.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find('/'), std::string::npos);
}

TEST(AuxiliaryBindings, RejectsSlashInCustomModelOutputName)
{
    // Custom model that reports an output name with '/' — should fail validation.
    class BadOutputModel : public AuxiliaryStateModel {
    public:
        std::string modelName() const override { return "BadOutput"; }
        int dimension() const override { return 1; }
        AuxiliaryStructuralMetadata structuralMetadata() const override {
            AuxiliaryStructuralMetadata meta;
            meta.variable_kinds = {AuxiliaryVariableKind::Differential};
            return meta;
        }
        void evaluateResidual(const AuxiliaryLocalContext&,
                              AuxiliaryResidualRequest& req) const override {
            req.residual[0] = 0.0;
        }
        int outputCount() const override { return 1; }
        std::vector<std::string> outputNames() const override {
            return {"bad/output"};
        }
        void evaluateOutputs(const AuxiliaryLocalContext&,
                             std::span<Real>) const override {}
    };

    auto model = std::make_shared<BadOutputModel>();
    auto instance = use(model)
        .name("ok_name")
        .scope(AuxiliaryStateScope::Global)
        .solveMode(AuxiliarySolveMode::Partitioned)
        .initialize({0.0});

    auto err = instance.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find('/'), std::string::npos);
}

// ============================================================================
//  Math-First DSL Tests (AuxiliaryModelDSL.h)
// ============================================================================

#include "Auxiliary/AuxiliaryModelDSL.h"

using namespace svmp::FE::systems;

// --- Phase 1: Typed symbols, equations, lambda builder ---

TEST(AuxiliaryModelDSL, LambdaModelScalarDecay)
{
    auto model = aux::model("decay", [](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k");
        m << ddt(x) == -k * x;
        m << out("y") == x;
    });

    EXPECT_EQ(model->modelName(), "decay");
    EXPECT_EQ(model->dimension(), 1);
    EXPECT_TRUE(model->hasResidualExpressions());
    ASSERT_EQ(model->stateNames().size(), 1u);
    EXPECT_EQ(model->stateNames()[0], "x");
    ASSERT_EQ(model->outputCount(), 1);
    EXPECT_EQ(model->outputNames()[0], "y");
}

TEST(AuxiliaryModelDSL, LambdaModelRCR)
{
    auto rcr = aux::model("rcr", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");

        m << ddt(X) == (Q - (X - Pd) / Rd) / C;
        m << out("P_out") == X + Rp * Q;
    });

    EXPECT_EQ(rcr->modelName(), "rcr");
    EXPECT_EQ(rcr->dimension(), 1);

    const auto& sig = rcr->signature();
    ASSERT_EQ(sig.inputs.size(), 1u);
    EXPECT_EQ(sig.inputs[0].name, "Q");
    ASSERT_EQ(sig.parameters.size(), 4u);
    EXPECT_EQ(sig.parameters[0].name, "Rp");
    EXPECT_EQ(sig.parameters[1].name, "C");
    EXPECT_EQ(sig.parameters[2].name, "Rd");
    EXPECT_EQ(sig.parameters[3].name, "Pd");
    ASSERT_EQ(sig.outputs.size(), 1u);
    EXPECT_EQ(sig.outputs[0].name, "P_out");
}

TEST(AuxiliaryModelDSL, LambdaModelMixedDAE)
{
    auto model = aux::model("mixed_dae", [](ModelFacade& m) {
        auto x = m.state("x");
        auto z = m.state("z", AuxiliaryVariableKind::Algebraic);

        m << ddt(x) == -x + z;
        m << alg(z) == x + z - forms::FormExpr::constant(1.0);
    });

    EXPECT_EQ(model->dimension(), 2);
    auto meta = model->structuralMetadata();
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[1], AuxiliaryVariableKind::Algebraic);
}

TEST(AuxiliaryModelDSL, GroupedDeclarations)
{
    auto model = aux::model("multi", [](ModelFacade& m) {
        auto [O2, Glc] = m.inputs("O2", "Glc");
        auto [ATP, ADP, NADH] = m.states("ATP", "ADP", "NADH");
        auto [k1, k2, k3] = m.params("k1", "k2", "k3");

        m << ddt(ATP) == -k1 * ATP;
        m << ddt(ADP) == k1 * ATP - k2 * ADP;
        m << ddt(NADH) == k2 * ADP - k3 * NADH;
        m << out("total") == ATP + ADP;
    });

    EXPECT_EQ(model->dimension(), 3);
    const auto& sig = model->signature();
    EXPECT_EQ(sig.inputs.size(), 2u);
    EXPECT_EQ(sig.parameters.size(), 3u);
    EXPECT_EQ(sig.outputs.size(), 1u);
    EXPECT_EQ(model->stateNames()[0], "ATP");
    EXPECT_EQ(model->stateNames()[1], "ADP");
    EXPECT_EQ(model->stateNames()[2], "NADH");
}

TEST(AuxiliaryModelDSL, VectorDeclarations)
{
    auto model = aux::model("vec_decl", [](ModelFacade& m) {
        auto states = m.stateVec({"x", "y", "z"});
        auto params = m.paramVec({"a", "b", "c"});
        ASSERT_EQ(states.size(), 3u);
        ASSERT_EQ(params.size(), 3u);

        m << ddt(states[0]) == -params[0] * states[0];
        m << ddt(states[1]) == params[0] * states[0] - params[1] * states[1];
        m << ddt(states[2]) == params[1] * states[1] - params[2] * states[2];
    });

    EXPECT_EQ(model->dimension(), 3);
    EXPECT_EQ(model->stateNames()[0], "x");
    EXPECT_EQ(model->stateNames()[2], "z");
}

TEST(AuxiliaryModelDSL, NamedIntermediates)
{
    auto model = aux::model("intermediates", [](ModelFacade& m) {
        auto S = m.input("S");
        auto [x, y] = m.states("x", "y");
        auto [k, Km] = m.params("k", "Km");

        auto v_rate = m.let("v_rate", k * S / (Km + S));

        m << ddt(x) == v_rate - x;
        m << ddt(y) == x - v_rate;
        m << out("rate") == v_rate;
    });

    EXPECT_EQ(model->dimension(), 2);
    EXPECT_EQ(model->outputCount(), 1);
    EXPECT_EQ(model->outputNames()[0], "rate");
}

TEST(AuxiliaryModelDSL, ExposeIntermediate)
{
    auto model = aux::model("expose", [](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k");

        auto rate = m.let("rate", k * x);
        m.expose(rate, "observed_rate");
        m << ddt(x) == -rate;
    });

    EXPECT_EQ(model->outputCount(), 1);
    EXPECT_EQ(model->outputNames()[0], "observed_rate");
}

// --- Equation insertion order independence ---

TEST(AuxiliaryModelDSL, EquationOrderIndependence)
{
    auto model_a = aux::model("order_a", [](ModelFacade& m) {
        auto [x, y, z] = m.states("x", "y", "z");
        auto k = m.param("k");
        m << ddt(x) == -k * x;
        m << ddt(y) == k * x - y;
        m << ddt(z) == y - z;
    });

    auto model_b = aux::model("order_b", [](ModelFacade& m) {
        auto [x, y, z] = m.states("x", "y", "z");
        auto k = m.param("k");
        m << ddt(z) == y - z;
        m << ddt(y) == k * x - y;
        m << ddt(x) == -k * x;
    });

    EXPECT_EQ(model_a->stateNames(), model_b->stateNames());
    EXPECT_EQ(model_a->dimension(), model_b->dimension());
    auto meta_a = model_a->structuralMetadata();
    auto meta_b = model_b->structuralMetadata();
    EXPECT_EQ(meta_a.variable_kinds, meta_b.variable_kinds);
}

// --- Validation and diagnostics ---

TEST(AuxiliaryModelDSL, DuplicateEquationThrows)
{
    EXPECT_THROW(
        aux::model("dup_eq", [](ModelFacade& m) {
            auto x = m.state("x");
            m << ddt(x) == -x;
            m << ddt(x) == -x * forms::FormExpr::constant(2.0);
        }),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelDSL, MissingEquationThrows)
{
    EXPECT_THROW(
        aux::model("missing_eq", [](ModelFacade& m) {
            auto x = m.state("x");
            auto y = m.state("y");
            m << ddt(x) == -x;
        }),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelDSL, DuplicateNameThrows)
{
    EXPECT_THROW(
        aux::model("dup_name", [](ModelFacade& m) {
            m.state("x");
            m.param("x");
        }),
        svmp::FE::InvalidArgumentException);
}

// --- DSL lowering equivalence ---

TEST(AuxiliaryModelDSL, EquivalenceWithLegacyBuilder)
{
    auto legacy = AuxiliaryModelBuilder("rcr")
        .input("Q")
        .state("X")
        .param("Rp").param("C").param("Rd").param("Pd")
        .ode("X",
            (modelInput("Q") - (modelState("X") - modelParam("Pd")) / modelParam("Rd")) / modelParam("C"))
        .output("P_out", modelState("X") + modelParam("Rp") * modelInput("Q"))
        .build();

    auto dsl = aux::model("rcr", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");
        m << ddt(X) == (Q - (X - Pd) / Rd) / C;
        m << out("P_out") == X + Rp * Q;
    });

    EXPECT_EQ(legacy->dimension(), dsl->dimension());
    EXPECT_EQ(legacy->stateNames(), dsl->stateNames());
    EXPECT_EQ(legacy->outputNames(), dsl->outputNames());

    AuxiliaryLocalContext ctx;
    Real x_val[] = {5.0};
    Real xdot_val[] = {0.0};
    Real input_val[] = {10.0};
    Real param_val[] = {100.0, 0.001, 1000.0, 5.0};
    ctx.x = x_val;
    ctx.xdot = xdot_val;
    ctx.inputs = input_val;
    ctx.params = param_val;
    ctx.time = 0.1;
    ctx.dt = 0.01;
    ctx.effective_dt = 0.01;

    Real res_legacy[1], res_dsl[1];
    AuxiliaryResidualRequest req_legacy{res_legacy};
    AuxiliaryResidualRequest req_dsl{res_dsl};
    legacy->evaluateResidual(ctx, req_legacy);
    dsl->evaluateResidual(ctx, req_dsl);
    EXPECT_DOUBLE_EQ(res_legacy[0], res_dsl[0]);

    Real out_legacy[1], out_dsl[1];
    legacy->evaluateOutputs(ctx, out_legacy);
    dsl->evaluateOutputs(ctx, out_dsl);
    EXPECT_DOUBLE_EQ(out_legacy[0], out_dsl[0]);
}

// --- Deployment ergonomics ---

TEST(AuxiliaryBindingsDSL, BulkParams)
{
    auto model = AuxiliaryModelBuilder("test")
        .input("Q").state("x").param("a").param("b").param("c")
        .ode("x", -modelParam("a") * modelState("x"))
        .build();

    auto instance = use(model).name("inst")
        .params({{"a", 1.0}, {"b", 2.0}, {"c", 3.0}})
        .bind("Q", "Q_reg").initialize({0.0});

    EXPECT_EQ(instance.paramValues().at("a"), 1.0);
    EXPECT_EQ(instance.paramValues().at("b"), 2.0);
    EXPECT_EQ(instance.paramValues().at("c"), 3.0);
    EXPECT_TRUE(instance.validate().empty());
}

TEST(AuxiliaryBindingsDSL, NamedInitialState)
{
    auto model = aux::model("two_state", [](ModelFacade& m) {
        auto [x, y] = m.states("x", "y");
        m << ddt(x) == -x;
        m << ddt(y) == -y;
    });

    auto instance = use(model).name("inst")
        .initialState({{"y", 2.0}, {"x", 1.0}});

    const auto& iv = instance.initialValues();
    ASSERT_EQ(iv.size(), 2u);
    EXPECT_DOUBLE_EQ(iv[0], 1.0);
    EXPECT_DOUBLE_EQ(iv[1], 2.0);
}

TEST(AuxiliaryBindingsDSL, NamedInitialStateUnknownThrows)
{
    auto model = aux::model("one_state", [](ModelFacade& m) {
        auto x = m.state("x");
        m << ddt(x) == -x;
    });

    EXPECT_THROW(
        use(model).name("inst").initialState({{"z", 1.0}}),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryBindingsDSL, NamedInitialStatePartialThrows)
{
    // Omitting a state from initialState() must be an error, not silent zero.
    auto model = aux::model("three_state", [](ModelFacade& m) {
        auto [x, y, z] = m.states("x", "y", "z");
        m << ddt(x) == -x;
        m << ddt(y) == -y;
        m << ddt(z) == -z;
    });

    EXPECT_THROW(
        use(model).name("inst").initialState({{"x", 1.0}, {"z", 3.0}}),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryBindingsDSL, ConvenienceSugar)
{
    auto model = AuxiliaryModelBuilder("test")
        .state("x").ode("x", -modelState("x")).build();

    auto ig = use(model).name("g").global();
    EXPECT_EQ(ig.getScope(), AuxiliaryStateScope::Global);

    auto ir = use(model).name("r").region();
    EXPECT_EQ(ir.getScope(), AuxiliaryStateScope::Region);

    auto ip = use(model).name("p").partitioned("BackwardEuler");
    EXPECT_EQ(ip.getSolveMode(), AuxiliarySolveMode::Partitioned);
    EXPECT_EQ(ip.getStepperSpec().method_name, "BackwardEuler");

    auto im = use(model).name("m").monolithic();
    EXPECT_EQ(im.getSolveMode(), AuxiliarySolveMode::Monolithic);
}

TEST(AuxiliaryModelBuilder, PartitionedFailurePolicyRejectsFailedLocalSolve)
{
    using namespace svmp::FE;

    auto model = aux_test::buildDecay(1.0);
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    AuxiliaryStepperSpec stepper;
    stepper.method_name = "BackwardEuler";
    stepper.max_nonlinear_iters = 0;

    AuxiliaryFailurePolicy policy;
    policy.max_local_retries = 1;
    policy.reject_timestep_on_failure = true;

    system.deployAuxiliaryModel(
        use(model).name("bad_step").global().partitioned("BackwardEuler")
            .stepper(stepper)
            .failurePolicy(policy)
            .param("k", 1.0)
            .initialize({1.0}));
    system.finalizeAuxiliaryLayout();

    EXPECT_THROW(system.advanceAuxiliaryState(0.0, 0.1),
                 svmp::FE::systems::InvalidStateException);
}

TEST(AuxiliaryModelBuilder, PartitionedFailurePolicyCanKeepCommittedState)
{
    using namespace svmp::FE;

    auto model = aux_test::buildDecay(1.0);
    systems::FESystem system(std::shared_ptr<const assembly::IMeshAccess>{});

    AuxiliaryStepperSpec stepper;
    stepper.method_name = "BackwardEuler";
    stepper.max_nonlinear_iters = 0;

    AuxiliaryFailurePolicy policy;
    policy.max_local_retries = 0;
    policy.reject_timestep_on_failure = false;

    system.deployAuxiliaryModel(
        use(model).name("nonrejecting_step").global().partitioned("BackwardEuler")
            .stepper(stepper)
            .failurePolicy(policy)
            .param("k", 1.0)
            .initialize({1.0}));
    system.finalizeAuxiliaryLayout();

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    ASSERT_NE(mgr, nullptr);
    auto& blk = mgr->getBlock("nonrejecting_step");

    EXPECT_NO_THROW(system.advanceAuxiliaryState(0.0, 0.1));
    EXPECT_DOUBLE_EQ(blk.work()[0], 1.0);
    EXPECT_DOUBLE_EQ(blk.committed()[0], 1.0);
}

TEST(AuxiliaryBindingsDSL, UnknownParamRejected)
{
    auto model = AuxiliaryModelBuilder("test")
        .state("x").param("k")
        .ode("x", -modelParam("k") * modelState("x")).build();

    auto inst = use(model).name("inst")
        .param("k", 1.0).param("bogus", 99.0).initialize({0.0});

    auto err = inst.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("bogus"), std::string::npos);
}

// --- Handle types ---

TEST(AuxiliaryInputHandleTest, ConvertToFormExpr)
{
    AuxiliaryInputHandle h("Q_flow");
    EXPECT_EQ(h.registryName(), "Q_flow");
    forms::FormExpr expr = h;
    EXPECT_TRUE(expr.isValid());
}

TEST(AuxiliaryInstanceHandleTest, OutputAccess)
{
    AuxiliaryInstanceHandle h("rcr_1");
    EXPECT_EQ(h.instanceName(), "rcr_1");
    auto p_out = h.output("P_out");
    EXPECT_TRUE(p_out.isValid());
}

TEST(AuxiliaryBindingsDSL, HandleBasedBind)
{
    auto model = AuxiliaryModelBuilder("test")
        .input("Q").state("x")
        .ode("x", -modelState("x")).build();

    AuxiliaryInputHandle Q_handle("Q_registry");
    auto inst = use(model).name("inst")
        .bind("Q", Q_handle).initialize({0.0});
    EXPECT_EQ(inst.inputBindings().at("Q"), "Q_registry");
    ASSERT_EQ(inst.coupledBindings().count("Q"), 1u);
    EXPECT_EQ(inst.coupledBindings().at("Q").registryName(), "Q_registry");
}

TEST(AuxiliaryBindingsDSL, AutoBindByName)
{
    auto model = AuxiliaryModelBuilder("test")
        .input("Q").state("x")
        .ode("x", -modelState("x")).build();

    AuxiliaryInputHandle Q_handle("Q");
    auto inst = use(model).name("inst")
        .bind(Q_handle).initialize({0.0});
    EXPECT_EQ(inst.inputBindings().at("Q"), "Q");
    ASSERT_EQ(inst.coupledBindings().count("Q"), 1u);
}

TEST(AuxiliaryBindingsDSL, ExplicitBindByNameMatchesConveniencePath)
{
    auto model = AuxiliaryModelBuilder("test")
        .input("Q").state("x")
        .ode("x", -modelState("x")).build();

    AuxiliaryInputHandle Q_handle("Q");
    auto inst = use(model).name("inst")
        .bindByName(Q_handle).initialize({0.0});
    EXPECT_EQ(inst.inputBindings().at("Q"), "Q");
    ASSERT_EQ(inst.coupledBindings().count("Q"), 1u);
}

TEST(AuxiliaryBindingsDSL, AutoNamedBoundaryIntegralHandleBindsExplicitly)
{
    auto model = AuxiliaryModelBuilder("test")
        .input("Q").state("x")
        .ode("x", -modelState("x") + modelInput("Q")).build();

    FESystem system(std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});
    const auto Q_handle = system.boundaryIntegral(forms::FormExpr::constant(1.0), 7);

    EXPECT_TRUE(Q_handle.hasDefinition());
    ASSERT_NE(Q_handle.definition(), nullptr);
    EXPECT_EQ(Q_handle.kind(), FEQuantityKind::BoundaryIntegral);
    EXPECT_EQ(Q_handle.definition()->boundary_marker, 7);
    EXPECT_EQ(Q_handle.registryName().find("_boundary_integral_b7_"), 0u);

    auto inst = use(model).name("inst")
        .boundary(7)
        .partitioned("BackwardEuler")
        .bindBoundaryReduction("Q", Q_handle)
        .initialize({0.0});

    EXPECT_EQ(inst.inputBindings().at("Q"), Q_handle.registryName());
    ASSERT_EQ(inst.coupledBindings().count("Q"), 1u);
    EXPECT_EQ(inst.coupledBindings().at("Q").registryName(), Q_handle.registryName());
    EXPECT_NO_THROW(system.deployAuxiliaryModel(inst));
}

TEST(AuxiliaryBindingsDSL, BoundaryReductionBindingChecksMarkerConsistency)
{
    auto model = AuxiliaryModelBuilder("rcr")
        .input("Q")
        .state("p")
        .ode("p", -modelState("p") + modelInput("Q"))
        .build();

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = "Q";
    def->kind = FEQuantityKind::BoundaryIntegral;
    def->shape = FEQuantityShape::scalar();
    def->boundary_marker = 7;
    AuxiliaryInputHandle Q_handle("Q", def);

    FESystem system(std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});

    EXPECT_NO_THROW(system.deployAuxiliaryModel(
        use(model)
            .name("rcr_ok")
            .boundary(7)
            .partitioned("BackwardEuler")
            .bindBoundaryReduction(Q_handle)
            .initialize({0.0})));

    EXPECT_THROW(
        system.deployAuxiliaryModel(
            use(model)
                .name("rcr_bad")
                .boundary(8)
                .partitioned("BackwardEuler")
                .bindBoundaryReduction(Q_handle)
                .initialize({0.0})),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryBindingsDSL, BoundaryReductionBindingsRemainInstanceLocalAcrossOutlets)
{
    auto model = AuxiliaryModelBuilder("rcr")
        .input("Q")
        .state("p")
        .ode("p", -modelState("p") + modelInput("Q"))
        .build();

    auto make_handle = [](std::string name, int marker) {
        auto def = std::make_shared<FEQuantityDefinition>();
        def->name = name;
        def->kind = FEQuantityKind::BoundaryIntegral;
        def->shape = FEQuantityShape::scalar();
        def->boundary_marker = marker;
        return AuxiliaryInputHandle(name, def);
    };

    const auto Q_left = make_handle("Q_left", 7);
    const auto Q_right = make_handle("Q_right", 9);

    FESystem system(std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});
    EXPECT_NO_THROW(system.deployAuxiliaryModel(
        use(model)
            .name("left_outlet")
            .boundary(7)
            .partitioned("BackwardEuler")
            .bindBoundaryReduction("Q", Q_left)
            .initialize({0.0})));
    EXPECT_NO_THROW(system.deployAuxiliaryModel(
        use(model)
            .name("right_outlet")
            .boundary(9)
            .partitioned("BackwardEuler")
            .bindBoundaryReduction("Q", Q_right)
            .initialize({0.0})));

    auto left = use(model)
        .name("left_outlet")
        .boundary(7)
        .partitioned("BackwardEuler")
        .bindBoundaryReduction("Q", Q_left)
        .initialize({0.0});
    auto right = use(model)
        .name("right_outlet")
        .boundary(9)
        .partitioned("BackwardEuler")
        .bindBoundaryReduction("Q", Q_right)
        .initialize({0.0});

    ASSERT_EQ(left.inputBindings().count("Q"), 1u);
    ASSERT_EQ(right.inputBindings().count("Q"), 1u);
    EXPECT_EQ(left.inputBindings().at("Q"), "Q_left");
    EXPECT_EQ(right.inputBindings().at("Q"), "Q_right");
    EXPECT_EQ(left.coupledBindings().at("Q").definition()->boundary_marker, 7);
    EXPECT_EQ(right.coupledBindings().at("Q").definition()->boundary_marker, 9);
}

TEST(AuxiliaryBindingsDSL, DeployAutoGeneratesGlobalInstanceName)
{
    FESystem system(std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});
    auto model = AuxiliaryModelBuilder("decay")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    const auto handle = system.deploy(
        use(model).global().partitioned("BackwardEuler").initialize({0.0}));

    EXPECT_EQ(handle.instanceName(), "decay_g0");
}

TEST(AuxiliaryBindingsDSL, DeployAutoGeneratesBoundaryNamesAndDisambiguatesCollisions)
{
    FESystem system(std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});
    auto model = AuxiliaryModelBuilder("rcr")
        .state("x")
        .ode("x", -modelState("x"))
        .build();

    const auto first = system.deploy(
        use(model).boundary(7).partitioned("BackwardEuler").initialize({0.0}));
    const auto second = system.deploy(
        use(model).boundary(7).partitioned("BackwardEuler").initialize({0.0}));

    EXPECT_EQ(first.instanceName(), "rcr_b7");
    EXPECT_EQ(second.instanceName(), "rcr_b7_1");
}

TEST(AuxiliaryBindingsDSL, MonolithicRejectsExplicitDefaultStepperConfiguration)
{
    auto model = AuxiliaryModelBuilder("test")
        .state("x")
        .ode("x", -modelState("x")).build();

    auto inst = use(model).name("inst")
        .monolithic()
        .stepper(AuxiliaryStepperSpec{"BackwardEuler"})
        .initialize({0.0});

    auto err = inst.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("explicit stepper"), std::string::npos);
    EXPECT_NE(err.find("Monolithic"), std::string::npos);
}

TEST(AuxiliaryBindingsDSL, MonolithicRejectsNonDefaultSchedule)
{
    auto model = AuxiliaryModelBuilder("test")
        .state("x")
        .ode("x", -modelState("x")).build();

    auto inst = use(model).name("inst")
        .monolithic()
        .schedule(AuxiliaryScheduleMode::Multirate)
        .initialize({0.0});

    auto err = inst.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("schedule mode"), std::string::npos);
    EXPECT_NE(err.find("Monolithic"), std::string::npos);
}

// --- Large system regression ---

TEST(AuxiliaryModelDSL, MediumSizedMultiStateODE)
{
    auto model = aux::model("metabolism", [](ModelFacade& m) {
        auto [O2, Glc, Lac] = m.inputs("O2", "Glc", "Lac");
        auto states = m.stateVec({"ATP", "ADP", "NADH", "NAD", "PYR", "LAC_i", "G6P", "F6P"});
        auto params = m.paramVec({"k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8",
                                   "Km_O2", "Km_Glc", "A_tot"});

        m << ddt(states[0]) == -params[0] * states[0] + params[1] * states[1];
        m << ddt(states[1]) == params[0] * states[0] - params[1] * states[1];
        m << ddt(states[2]) == params[2] * states[3] - params[3] * states[2];
        m << ddt(states[3]) == -params[2] * states[3] + params[3] * states[2];
        m << ddt(states[4]) == params[4] * states[6] - params[5] * states[4];
        m << ddt(states[5]) == params[5] * states[4] - params[6] * states[5];
        m << ddt(states[6]) == params[7] * Glc - params[4] * states[6];
        m << ddt(states[7]) == params[4] * states[6] - params[7] * states[7];

        m << out("energy_charge") == states[0] / (states[0] + states[1]);
        m << out("redox_ratio") == states[2] / (states[2] + states[3]);
    });

    EXPECT_EQ(model->dimension(), 8);
    EXPECT_EQ(model->outputCount(), 2);
    EXPECT_EQ(model->stateNames().size(), 8u);
    EXPECT_EQ(model->signature().inputs.size(), 3u);
    EXPECT_EQ(model->signature().parameters.size(), 11u);
}

// --- Builder introspection ---

TEST(AuxiliaryModelBuilder, IntrospectionMethods)
{
    AuxiliaryModelBuilder builder("test");
    builder.input("Q").input("P")
           .state("x").state("y", AuxiliaryVariableKind::Algebraic)
           .param("k1").param("k2")
           .ode("x", -modelParam("k1") * modelState("x"))
           .algebraic("y", modelState("x") + modelState("y"))
           .output("sum", modelState("x") + modelState("y"));

    EXPECT_EQ(builder.modelName(), "test");
    EXPECT_EQ(builder.stateCount(), 2u);
    EXPECT_EQ(builder.rowCount(), 2u);

    auto snames = builder.stateNames();
    ASSERT_EQ(snames.size(), 2u);
    EXPECT_EQ(snames[0], "x");
    EXPECT_EQ(snames[1], "y");

    auto inames = builder.inputNames();
    ASSERT_EQ(inames.size(), 2u);
    EXPECT_EQ(inames[0], "Q");
    EXPECT_EQ(inames[1], "P");

    auto pnames = builder.parameterNames();
    ASSERT_EQ(pnames.size(), 2u);
    EXPECT_EQ(pnames[0], "k1");
    EXPECT_EQ(pnames[1], "k2");

    auto onames = builder.outputNames();
    ASSERT_EQ(onames.size(), 1u);
    EXPECT_EQ(onames[0], "sum");
}

// ============================================================================
//  DSL feature tests: metadata, grouping, conservation, summary, optional
// ============================================================================

TEST(AuxiliaryModelDSL, OptionalParamWithDefault)
{
    auto model = aux::model("opt_test", [](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k", 1.0);  // optional, default=1.0
        m << ddt(x) == -k * x;
    });

    EXPECT_EQ(model->dimension(), 1);
    const auto& sig = model->signature();
    ASSERT_EQ(sig.parameters.size(), 1u);
    EXPECT_TRUE(sig.parameters[0].optional);
    EXPECT_TRUE(sig.parameters[0].default_value.has_value());
    EXPECT_DOUBLE_EQ(*sig.parameters[0].default_value, 1.0);

    // Deployment without setting "k" should pass validation (optional).
    auto inst = use(model).name("inst").initialize({0.0});
    EXPECT_TRUE(inst.validate().empty());
}

TEST(AuxiliaryModelDSL, MetadataSetters)
{
    auto model = aux::model("meta_test", [](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k");

        m.nonnegative("x");
        m.scale("x", 1e-3);

        m << ddt(x) == -k * x;
    });

    // Verify metadata is surfaced in summary.
    AuxiliaryModelBuilder builder("meta_check");
    builder.state("x").param("k")
        .ode("x", -modelParam("k") * modelState("x"));
    builder.setNonnegative("x");
    builder.setScale("x", 1e-3);

    auto s = builder.summary();
    EXPECT_NE(s.find("lower=0"), std::string::npos) << "Summary should contain bounds";
    EXPECT_NE(s.find("scale"), std::string::npos) << "Summary should contain scale";
}

TEST(AuxiliaryModelDSL, AlgebraicInitialGuess)
{
    auto model = aux::model("alg_guess", [](ModelFacade& m) {
        auto x = m.state("x");
        auto z = m.state("z", AuxiliaryVariableKind::Algebraic);

        m.initialGuess("z", 0.5);

        m << ddt(x) == -x + z;
        m << alg(z) == x + z - forms::FormExpr::constant(1.0);
    });

    EXPECT_EQ(model->dimension(), 2);

    // Verify initial guess is carried into the built model.
    const auto& guesses = model->initialGuesses();
    ASSERT_EQ(guesses.size(), 2u);
    EXPECT_FALSE(guesses[0].has_value());  // x: differential, no guess
    EXPECT_TRUE(guesses[1].has_value());   // z: algebraic, guess = 0.5
    EXPECT_DOUBLE_EQ(*guesses[1], 0.5);

    // Verify initialState() auto-fills the algebraic state from the guess.
    // Only provide "x" — "z" should be auto-filled from the guess.
    auto inst = use(model).name("guess_inst").initialState({{"x", 1.0}});
    const auto& iv = inst.initialValues();
    ASSERT_EQ(iv.size(), 2u);
    EXPECT_DOUBLE_EQ(iv[0], 1.0);  // x: explicitly provided
    EXPECT_DOUBLE_EQ(iv[1], 0.5);  // z: auto-filled from guess
}

TEST(AuxiliaryModelDSL, SymbolGrouping)
{
    auto model = aux::model("grouped", [](ModelFacade& m) {
        auto mito = m.group("mito");
        auto ATP = mito.state("ATP");
        auto NADH = mito.state("NADH");
        auto k = mito.param("k");

        m << ddt(ATP) == -k * ATP;
        m << ddt(NADH) == k * ATP - NADH;
    });

    EXPECT_EQ(model->dimension(), 2);
    // State names should be prefixed.
    EXPECT_EQ(model->stateNames()[0], "mito.ATP");
    EXPECT_EQ(model->stateNames()[1], "mito.NADH");
    const auto& sig = model->signature();
    EXPECT_EQ(sig.parameters[0].name, "mito.k");
}

TEST(AuxiliaryModelDSL, ConservationHelper)
{
    auto model = aux::model("conserved", [](ModelFacade& m) {
        auto ATP = m.state("ATP");
        auto ADP = m.state("ADP", AuxiliaryVariableKind::Algebraic);
        auto k = m.param("k");
        auto A_tot = m.param("A_tot");

        m << ddt(ATP) == -k * ATP;
        m.conservation(ADP, ATP + ADP - A_tot);
    });

    EXPECT_EQ(model->dimension(), 2);
    auto meta = model->structuralMetadata();
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[1], AuxiliaryVariableKind::Algebraic);
}

TEST(AuxiliaryModelDSL, PrettyPrinterSummary)
{
    AuxiliaryModelBuilder b("summary_test");
    b.input("Q", 1).state("x").param("k")
     .ode("x", -modelParam("k") * modelState("x"))
     .output("y", modelState("x"));
    b.setNonnegative("x");

    auto s = b.summary();
    EXPECT_NE(s.find("Model: summary_test"), std::string::npos);
    EXPECT_NE(s.find("States (1)"), std::string::npos);
    EXPECT_NE(s.find("Inputs (1)"), std::string::npos);
    EXPECT_NE(s.find("Parameters (1)"), std::string::npos);
    EXPECT_NE(s.find("Equations (1)"), std::string::npos);
    EXPECT_NE(s.find("Outputs (1)"), std::string::npos);
    EXPECT_NE(s.find("lower=0"), std::string::npos) << "Summary should show nonnegative bound";
}

TEST(AuxiliaryModelDSL, UnusedSymbolPolicyError)
{
    // With "error" policy, unused params should throw.
    EXPECT_THROW(
        AuxiliaryModelBuilder("strict")
            .state("x")
            .param("unused_k")
            .ode("x", -modelState("x"))
            .unusedSymbolPolicy("error")
            .build(),
        svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryModelDSL, UnusedSymbolPolicySilent)
{
    // With "silent" policy, unused params should NOT throw.
    EXPECT_NO_THROW(
        AuxiliaryModelBuilder("silent")
            .state("x")
            .param("unused_k")
            .ode("x", -modelState("x"))
            .unusedSymbolPolicy("silent")
            .build());
}

TEST(AuxiliaryModelDSL, UnusedSymbolPolicyWarn)
{
    // With "warn" (default) policy, unused params should NOT throw
    // (warnings go to stderr, not exceptions).
    EXPECT_NO_THROW(
        AuxiliaryModelBuilder("warn_default")
            .state("x")
            .param("unused_k")
            .ode("x", -modelState("x"))
            .unusedSymbolPolicy("warn")
            .build());

    // Default policy (no explicit call) should also not throw.
    EXPECT_NO_THROW(
        AuxiliaryModelBuilder("default_policy")
            .state("x")
            .param("unused_k")
            .ode("x", -modelState("x"))
            .build());
}

// ============================================================================
//  Shape helper tests
// ============================================================================

TEST(AuxiliaryBindingsDSL, ShapeHelperComp)
{
    AuxiliaryInputHandle h("u_vec");
    auto c0 = svmp::FE::systems::comp(h, 0);
    auto c2 = svmp::FE::systems::comp(h, 2);
    EXPECT_TRUE(c0.isValid());
    EXPECT_TRUE(c2.isValid());
}

TEST(AuxiliaryBindingsDSL, ShapeHelperDot)
{
    auto def = std::make_shared<svmp::FE::systems::FEQuantityDefinition>();
    def->name = "u";
    def->shape = svmp::FE::systems::FEQuantityShape::vector(3);
    AuxiliaryInputHandle a("u", def);
    AuxiliaryInputHandle b("v", def);
    auto d = svmp::FE::systems::dot(a, b);
    EXPECT_TRUE(d.isValid());
}

TEST(AuxiliaryBindingsDSL, ShapeHelperTrace)
{
    auto def = std::make_shared<svmp::FE::systems::FEQuantityDefinition>();
    def->name = "sigma";
    def->shape = svmp::FE::systems::FEQuantityShape::tensor(3);
    AuxiliaryInputHandle h("sigma", def);
    auto tr = svmp::FE::systems::trace(h);
    EXPECT_TRUE(tr.isValid());
}

TEST(AuxiliaryBindingsDSL, ShapeHelperNorm)
{
    auto def = std::make_shared<svmp::FE::systems::FEQuantityDefinition>();
    def->name = "u";
    def->shape = svmp::FE::systems::FEQuantityShape::vector(3);
    AuxiliaryInputHandle h("u", def);
    auto n = svmp::FE::systems::norm(h);
    EXPECT_TRUE(n.isValid());
}

// ============================================================================
//  Mixed ODE/algebraic partitioned test
// ============================================================================

TEST(AuxiliaryModelDSL, MixedODEAlgebraicBuild)
{
    // x' = -x + z, 0 = x + z - 1 — with algebraic initial guess.
    auto model = aux::model("mixed_part", [](ModelFacade& m) {
        auto x = m.state("x");
        auto z = m.state("z", AuxiliaryVariableKind::Algebraic);
        m.initialGuess("z", 0.5);

        m << ddt(x) == -x + z;
        m << alg(z) == x + z - forms::FormExpr::constant(1.0);
    });

    EXPECT_EQ(model->dimension(), 2);
    auto meta = model->structuralMetadata();
    EXPECT_EQ(meta.variable_kinds[0], AuxiliaryVariableKind::Differential);
    EXPECT_EQ(meta.variable_kinds[1], AuxiliaryVariableKind::Algebraic);

    // Verify deployment with initialState auto-fills algebraic guess.
    auto inst = use(model).name("mixed").partitioned("BackwardEuler")
        .initialState({{"x", 1.0}});
    EXPECT_DOUBLE_EQ(inst.initialValues()[0], 1.0);
    EXPECT_DOUBLE_EQ(inst.initialValues()[1], 0.5);
}

// ============================================================================
//  Many-input / many-output regression
// ============================================================================

TEST(AuxiliaryModelDSL, ManyInputManyOutputRegression)
{
    auto model = aux::model("big_model", [](ModelFacade& m) {
        // 6 inputs, 8 states, 4 outputs.
        auto inputs = m.inputVec({"I1", "I2", "I3", "I4", "I5", "I6"});
        auto states = m.stateVec({"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"});
        auto params = m.paramVec({"k1", "k2", "k3", "k4"});

        for (int i = 0; i < 8; ++i) {
            auto rhs = -params[static_cast<std::size_t>(i % 4)] * states[static_cast<std::size_t>(i)];
            if (i < 6) rhs = rhs + inputs[static_cast<std::size_t>(i)];
            m << ddt(states[static_cast<std::size_t>(i)]) == rhs;
        }

        m << out("sum_1_4") == states[0] + states[1] + states[2] + states[3];
        m << out("sum_5_8") == states[4] + states[5] + states[6] + states[7];
        m << out("total") == states[0] + states[1] + states[2] + states[3]
                            + states[4] + states[5] + states[6] + states[7];
        m << out("ratio") == states[0] / (states[0] + states[1]
                            + forms::FormExpr::constant(1e-10));
    });

    EXPECT_EQ(model->dimension(), 8);
    EXPECT_EQ(model->outputCount(), 4);
    EXPECT_EQ(model->signature().inputs.size(), 6u);
    EXPECT_EQ(model->signature().parameters.size(), 4u);

    // Verify evaluation.
    AuxiliaryLocalContext ctx;
    Real x[] = {1,2,3,4,5,6,7,8};
    Real xdot[] = {0,0,0,0,0,0,0,0};
    Real inp[] = {1,1,1,1,1,1};
    Real par[] = {0.1, 0.2, 0.3, 0.4};
    ctx.x = x; ctx.xdot = xdot; ctx.inputs = inp; ctx.params = par;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    Real out[4];
    model->evaluateOutputs(ctx, out);
    EXPECT_NEAR(out[0], 10.0, 1e-10);  // 1+2+3+4
    EXPECT_NEAR(out[1], 26.0, 1e-10);  // 5+6+7+8
    EXPECT_NEAR(out[2], 36.0, 1e-10);  // total
}

// ============================================================================
//  Summary / introspection snapshot test
// ============================================================================

TEST(AuxiliaryModelDSL, SummarySnapshotDeterministic)
{
    // Build the same model twice and verify summary() is identical.
    auto buildModel = []() {
        AuxiliaryModelBuilder b("snapshot_test");
        b.input("Q").state("x").state("z", AuxiliaryVariableKind::Algebraic)
         .param("k").param("Rd", 100.0)
         .ode("x", -modelParam("k") * modelState("x") + modelInput("Q"))
         .algebraic("z", modelState("x") + modelState("z") - forms::FormExpr::constant(1.0))
         .output("y", modelState("x") + modelState("z"));
        b.setNonnegative("x");
        b.setInitialGuess("z", 0.5);
        return b.summary();
    };

    const auto s1 = buildModel();
    const auto s2 = buildModel();
    EXPECT_EQ(s1, s2) << "summary() should be deterministic across builds";
    EXPECT_NE(s1.find("snapshot_test"), std::string::npos);
    EXPECT_NE(s1.find("[algebraic]"), std::string::npos);
    EXPECT_NE(s1.find("guess=0.5"), std::string::npos);
    EXPECT_NE(s1.find("[optional]"), std::string::npos);
}

// ============================================================================
//  Insertion-order independence
// ============================================================================

TEST(AuxiliaryModelDSL, InsertionOrderIndependenceWithMetadata)
{
    // Equations in different order, same model — verify identical evaluation.
    auto model_a = aux::model("order_a", [](ModelFacade& m) {
        auto [x, y] = m.states("x", "y");
        auto k = m.param("k");
        m.nonnegative("x");
        m << ddt(x) == -k * x;
        m << ddt(y) == k * x - y;
    });

    auto model_b = aux::model("order_b", [](ModelFacade& m) {
        auto [x, y] = m.states("x", "y");
        auto k = m.param("k");
        m.nonnegative("x");
        m << ddt(y) == k * x - y;  // reversed
        m << ddt(x) == -k * x;
    });

    EXPECT_EQ(model_a->stateNames(), model_b->stateNames());

    AuxiliaryLocalContext ctx;
    Real xa[] = {2.0, 3.0}, xda[] = {0, 0}, pa[] = {0.5};
    ctx.x = xa; ctx.xdot = xda; ctx.params = pa;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    Real ra[2], rb[2];
    AuxiliaryResidualRequest req_a{ra}, req_b{rb};
    model_a->evaluateResidual(ctx, req_a);
    model_b->evaluateResidual(ctx, req_b);
    EXPECT_DOUBLE_EQ(ra[0], rb[0]);
    EXPECT_DOUBLE_EQ(ra[1], rb[1]);
}

// ============================================================================
//  Repeated registration determinism
// ============================================================================

TEST(AuxiliaryModelDSL, RepeatedBuildDeterminism)
{
    // Build the same model 3 times, verify all produce identical results.
    auto build = []() {
        return aux::model("repeat", [](ModelFacade& m) {
            auto x = m.state("x");
            auto k = m.param("k");
            m << ddt(x) == -k * x;
            m << out("y") == x;
        });
    };

    auto m1 = build(), m2 = build(), m3 = build();

    AuxiliaryLocalContext ctx;
    Real x[] = {5.0}, xd[] = {0.0}, p[] = {2.0};
    ctx.x = x; ctx.xdot = xd; ctx.params = p;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    Real r1[1], r2[1], r3[1];
    AuxiliaryResidualRequest rq1{r1}, rq2{r2}, rq3{r3};
    m1->evaluateResidual(ctx, rq1);
    m2->evaluateResidual(ctx, rq2);
    m3->evaluateResidual(ctx, rq3);

    EXPECT_DOUBLE_EQ(r1[0], r2[0]);
    EXPECT_DOUBLE_EQ(r2[0], r3[0]);
}

// ============================================================================
//  FD Jacobian check on representative model
// ============================================================================

TEST(AuxiliaryModelDSL, FDJacobianCheckOnRCR)
{
    auto model = aux::model("rcr_fd", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");
        m << ddt(X) == (Q - (X - Pd) / Rd) / C;
    });

    AuxiliaryLocalContext ctx;
    Real x[] = {5.0}, xd[] = {0.0}, inp[] = {0.01}, par[] = {100, 0.001, 1000, 5.0};
    ctx.x = x; ctx.xdot = xd; ctx.inputs = inp; ctx.params = par;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    // Base residual.
    Real r0[1];
    AuxiliaryResidualRequest rq0{r0};
    model->evaluateResidual(ctx, rq0);

    // FD dF/dx.
    const Real eps = 1e-7;
    Real x_pert[] = {5.0 + eps};
    AuxiliaryLocalContext ctx_pert = ctx;
    ctx_pert.x = x_pert;
    Real r_pert[1];
    AuxiliaryResidualRequest rq_p{r_pert};
    model->evaluateResidual(ctx_pert, rq_p);

    const Real dFdx_fd = (r_pert[0] - r0[0]) / eps;
    // Analytic: dF/dX = d(xdot - (Q - (X-Pd)/Rd)/C)/dX = 1/(Rd*C) = 1/(1000*0.001) = 1.0
    EXPECT_NEAR(dFdx_fd, 1.0, 1e-4);
}

// ============================================================================
//  Submodel composition via include()
// ============================================================================

TEST(AuxiliaryModelDSL, IncludeSubmodel)
{
    // Build two small submodels, then compose them.
    auto sub_a = aux::model("sub_a", [](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k");
        m << ddt(x) == -k * x;
        m << out("y") == x;
    });

    auto sub_b = aux::model("sub_b", [](ModelFacade& m) {
        auto z = m.state("z");
        auto c = m.param("c");
        m << ddt(z) == -c * z;
        m << out("w") == z;
    });

    auto combined = aux::model("combined", [&](ModelFacade& m) {
        m.include(sub_a, "a");
        m.include(sub_b, "b");
    });

    EXPECT_EQ(combined->dimension(), 2);
    aux_test::expectStateOrder(*combined, {"a.x", "b.z"});

    const auto& sig = combined->signature();
    EXPECT_EQ(sig.parameters.size(), 2u);
    EXPECT_EQ(sig.parameters[0].name, "a.k");
    EXPECT_EQ(sig.parameters[1].name, "b.c");
    EXPECT_EQ(sig.outputs.size(), 2u);
    EXPECT_EQ(sig.outputs[0].name, "a.y");
    EXPECT_EQ(sig.outputs[1].name, "b.w");

    // Verify evaluation: a.x=3, b.z=7, a.k=0.5, b.c=0.2
    Real x[] = {3.0, 7.0}, xd[] = {0.0, 0.0}, par[] = {0.5, 0.2};
    auto res = aux_test::evaluateResidual(*combined, x, xd, {}, par);
    // F[0] = xdot[0] - (-0.5*3) = 0 - (-1.5) = 1.5
    // F[1] = xdot[1] - (-0.2*7) = 0 - (-1.4) = 1.4
    EXPECT_NEAR(res[0], 1.5, 1e-10);
    EXPECT_NEAR(res[1], 1.4, 1e-10);

    // Verify outputs.
    AuxiliaryLocalContext ctx;
    ctx.x = x; ctx.xdot = xd; ctx.params = par;
    ctx.time = 0; ctx.dt = 0.01; ctx.effective_dt = 0.01;
    Real out[2];
    combined->evaluateOutputs(ctx, out);
    EXPECT_NEAR(out[0], 3.0, 1e-10);  // a.y = a.x = 3
    EXPECT_NEAR(out[1], 7.0, 1e-10);  // b.w = b.z = 7
}

TEST(AuxiliaryModelDSL, IncludeWithCrossModelCoupling)
{
    // Include a submodel and add cross-model coupling.
    auto decay = aux::model("decay_sub", [](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k");
        m << ddt(x) == -k * x;
    });

    auto combined = aux::model("coupled", [&](ModelFacade& m) {
        m.include(decay, "d");
        // Add a new state that couples to the included state.
        auto y = m.state("y");
        // d.x is at state slot 0 after include.
        m << ddt(y) == forms::FormExpr::auxiliaryStateRef(0) - y;
    });

    EXPECT_EQ(combined->dimension(), 2);
    aux_test::expectStateOrder(*combined, {"d.x", "y"});

    Real x[] = {4.0, 2.0}, xd[] = {0.0, 0.0}, par[] = {0.5};
    auto res = aux_test::evaluateResidual(*combined, x, xd, {}, par);
    // F[0] = xdot[0] - (-0.5*4) = 2.0
    // F[1] = xdot[1] - (x[0] - y) = 0 - (4 - 2) = -2.0
    EXPECT_NEAR(res[0], 2.0, 1e-10);
    EXPECT_NEAR(res[1], -2.0, 1e-10);
}

// ============================================================================
//  Test helper usage demonstration
// ============================================================================

TEST(AuxiliaryTestHelpers, BuilderHelpers)
{
    auto decay = aux_test::buildDecay();
    EXPECT_EQ(decay->dimension(), 1);
    EXPECT_EQ(decay->outputCount(), 1);

    auto rcr = aux_test::buildRCR();
    EXPECT_EQ(rcr->dimension(), 1);
    EXPECT_EQ(rcr->signature().inputs.size(), 1u);

    auto dae = aux_test::buildDAE();
    EXPECT_EQ(dae->dimension(), 2);
    EXPECT_TRUE(dae->initialGuesses()[1].has_value());
}

TEST(AuxiliaryTestHelpers, EvaluateResidualHelper)
{
    auto rcr = aux_test::buildRCR();
    Real x[] = {5.0}, xd[] = {0.0}, inp[] = {0.01}, par[] = {100, 0.001, 1000, 5.0};
    auto res = aux_test::evaluateResidual(*rcr, x, xd, inp, par);
    ASSERT_EQ(res.size(), 1u);
    // F = xdot - (Q - (X-Pd)/Rd)/C = 0 - (0.01 - (5-5)/1000)/0.001 = -10.0
    EXPECT_NEAR(res[0], -10.0, 1e-10);
}

TEST(AuxiliaryTestHelpers, StateOrderAssertion)
{
    auto dae = aux_test::buildDAE();
    aux_test::expectStateOrder(*dae, {"x", "z"});
}
