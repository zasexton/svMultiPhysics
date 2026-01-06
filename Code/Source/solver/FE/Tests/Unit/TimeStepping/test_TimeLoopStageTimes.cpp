/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Core/Types.h"
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

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace ts_test = svmp::FE::timestepping::test;

namespace {

struct ContextSnapshot {
    double time{0.0};
    double dt{0.0};
    double time_weight{1.0};
    double non_time_weight{1.0};
    std::vector<double> dt1{};
};

bool sameSnapshot(const ContextSnapshot& a, const ContextSnapshot& b, double tol)
{
    if (std::abs(a.time - b.time) > tol) return false;
    if (std::abs(a.dt - b.dt) > tol) return false;
    if (std::abs(a.time_weight - b.time_weight) > tol) return false;
    if (std::abs(a.non_time_weight - b.non_time_weight) > tol) return false;
    if (a.dt1.size() != b.dt1.size()) return false;
    for (std::size_t i = 0; i < a.dt1.size(); ++i) {
        if (std::abs(a.dt1[i] - b.dt1[i]) > 1e-12) return false;
    }
    return true;
}

struct RecordedContexts {
    std::vector<ContextSnapshot> snapshots{};
};

class RecordingKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    RecordingKernel(std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner, RecordedContexts* out)
        : inner_(std::move(inner))
        , out_(out)
    {
        if (!inner_) {
            throw std::runtime_error("RecordingKernel: inner is null");
        }
        if (!out_) {
            throw std::runtime_error("RecordingKernel: out is null");
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

        const auto* ti = ctx.timeIntegrationContext();
        if (!ti) {
            return;
        }

        ContextSnapshot snap;
        snap.time = static_cast<double>(ctx.time());
        snap.dt = static_cast<double>(ctx.timeStep());
        snap.time_weight = static_cast<double>(ti->time_derivative_term_weight);
        snap.non_time_weight = static_cast<double>(ti->non_time_derivative_term_weight);
        if (ti->dt1) {
            snap.dt1.reserve(ti->dt1->a.size());
            for (const auto coeff : ti->dt1->a) {
                snap.dt1.push_back(static_cast<double>(coeff));
            }
        }

        constexpr double tol = 1e-15;
        for (const auto& existing : out_->snapshots) {
            if (sameSnapshot(existing, snap, tol)) {
                return;
            }
        }
        out_->snapshots.push_back(std::move(snap));
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
        return "Recording(" + inner_->name() + ")";
    }

private:
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner_{};
    RecordedContexts* out_{nullptr};
};

struct RunResult {
    RecordedContexts ctxs;
    std::vector<Real> final_u;
};

RunResult runOneStep(svmp::FE::timestepping::SchemeKind scheme,
                     double dt,
                     double lambda,
                     std::function<void(svmp::FE::timestepping::TimeLoopOptions&)> configure_opts = {})
{
    RunResult out;

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
    auto inner_kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    auto kernel = std::make_shared<RecordingKernel>(inner_kernel, &out.ctxs);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);
    EXPECT_TRUE(sys.isSetup());

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return out;
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    EXPECT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/3);
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    for (int k = 1; k <= history.historyDepth(); ++k) {
        ts_test::setVectorByDof(history.uPrevK(k), u0);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = dt;
    opts.dt = dt;
    opts.max_steps = 10;
    opts.scheme = scheme;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 10;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;
    if (configure_opts) {
        configure_opts(opts);
    }

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history);
    EXPECT_TRUE(rep.success) << rep.message;
    EXPECT_NEAR(rep.final_time, dt, 1e-12);

    out.final_u = ts_test::getVectorByDof(history.uPrev());
    return out;
}

const ContextSnapshot* findSnapshot(const RecordedContexts& ctxs,
                                    double time,
                                    double time_weight,
                                    double non_time_weight,
                                    double tol = 1e-14)
{
    for (const auto& s : ctxs.snapshots) {
        if (std::abs(s.time - time) > tol) continue;
        if (std::abs(s.time_weight - time_weight) > tol) continue;
        if (std::abs(s.non_time_weight - non_time_weight) > tol) continue;
        return &s;
    }
    return nullptr;
}

} // namespace

TEST(TimeLoopStageTimes, GeneralizedAlphaUsesAlphaFStageTime)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double dt = 0.1;
    const double lambda = 1.0;

    auto res = runOneStep(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                          dt,
                          lambda,
                          [](svmp::FE::timestepping::TimeLoopOptions& o) {
                              // rho_inf=1 => alpha_f=1/(1+rho_inf)=0.5 (1st-order generalized-alpha)
                              o.generalized_alpha_rho_inf = 1.0;
                          });

    ASSERT_FALSE(res.ctxs.snapshots.empty());
    const double expected_stage_time = 0.5 * dt;

    const bool saw_stage_time = std::any_of(res.ctxs.snapshots.begin(),
                                            res.ctxs.snapshots.end(),
                                            [&](const ContextSnapshot& s) {
                                                return std::abs(s.time - expected_stage_time) <= 1e-14;
                                            });
    EXPECT_TRUE(saw_stage_time);
}

TEST(TimeLoopStageTimes, TRBDF2UsesExpectedStageTimesAndWeights)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double dt = 0.2;
    const double gamma = 0.25;
    const double lambda = 1.0;
    const double dt1 = gamma * dt;
    const double dt2 = dt - dt1;

    auto res = runOneStep(svmp::FE::timestepping::SchemeKind::TRBDF2,
                          dt,
                          lambda,
                          [&](svmp::FE::timestepping::TimeLoopOptions& o) {
                              o.trbdf2_gamma = gamma;
                          });

    ASSERT_FALSE(res.ctxs.snapshots.empty());

    // Stage 1 TR split: current non-time term weight is 0.5 at t + dt1.
    const auto* stage1 = findSnapshot(res.ctxs, /*time=*/dt1, /*time_weight=*/1.0, /*non_time_weight=*/0.5);
    ASSERT_NE(stage1, nullptr);
    ASSERT_EQ(stage1->dt1.size(), 2u);
    EXPECT_NEAR(stage1->dt, dt1, 1e-14);
    EXPECT_NEAR(stage1->dt1[0], 1.0 / dt1, 1e-14);
    EXPECT_NEAR(stage1->dt1[1], -1.0 / dt1, 1e-14);

    // Stage 2 BDF2 step over dt2 on [t+dt1, t+dt]: dt1 weights should match the variable-step BDF2 stencil.
    const auto* stage2 = findSnapshot(res.ctxs, /*time=*/dt, /*time_weight=*/1.0, /*non_time_weight=*/1.0);
    ASSERT_NE(stage2, nullptr);
    ASSERT_EQ(stage2->dt1.size(), 3u);
    EXPECT_NEAR(stage2->dt, dt2, 1e-14);

    const double r = dt2 / dt1;
    const double inv_dt2 = 1.0 / dt2;
    EXPECT_NEAR(stage2->dt1[0], ((1.0 + 2.0 * r) / (1.0 + r)) * inv_dt2, 1e-12);
    EXPECT_NEAR(stage2->dt1[1], (-(1.0 + r)) * inv_dt2, 1e-12);
    EXPECT_NEAR(stage2->dt1[2], ((r * r) / (1.0 + r)) * inv_dt2, 1e-12);
}
