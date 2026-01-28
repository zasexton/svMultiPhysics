/**
 * @file test_TimeIntegrator.cpp
 * @brief Unit tests for Systems TimeIntegrator implementations
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "Assembly/GlobalSystemView.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::TimeIntegrationContext;
using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::BDF2Integrator;
using svmp::FE::systems::BDFIntegrator;
using svmp::FE::systems::BackwardDifferenceIntegrator;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::SystemStateView;
using svmp::FE::systems::TransientSystem;

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

void expectStencilNear(const std::vector<Real>& got,
                       const std::vector<Real>& expected,
                       Real tol)
{
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], tol);
    }
}

std::vector<std::span<const Real>> makeHistorySpans(const std::vector<std::vector<Real>>& storage)
{
    std::vector<std::span<const Real>> spans;
    spans.reserve(storage.size());
    for (const auto& v : storage) {
        spans.push_back(v);
    }
    return spans;
}

} // namespace

TEST(TimeIntegrator, BDFIntegrator_Order1_MatchesBackwardEuler)
{
    const double dt = 0.5;
    SystemStateView state;
    state.dt = dt;

    std::vector<std::vector<Real>> hist_storage = {{Real(0.0)}};
    auto hist_spans = makeHistorySpans(hist_storage);
    state.u_history = hist_spans;

    BackwardDifferenceIntegrator be;
    const auto ctx_be = be.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx_be.dt1.has_value());

    BDFIntegrator bdf1(1);
    const auto ctx_bdf = bdf1.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx_bdf.dt1.has_value());

    expectStencilNear(ctx_bdf.dt1->a, ctx_be.dt1->a, 1e-12);
}

TEST(TimeIntegrator, BDFIntegrator_Order2_MatchesBDF2)
{
    const double dt = 0.25;
    SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt;

    std::vector<std::vector<Real>> hist_storage = {{Real(0.0)}, {Real(0.0)}};
    auto hist_spans = makeHistorySpans(hist_storage);
    state.u_history = hist_spans;

    BDF2Integrator bdf2;
    const auto ctx_bdf2 = bdf2.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx_bdf2.dt1.has_value());

    BDFIntegrator bdf(2);
    const auto ctx_bdf = bdf.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx_bdf.dt1.has_value());

    expectStencilNear(ctx_bdf.dt1->a, ctx_bdf2.dt1->a, 1e-12);
}

TEST(TimeIntegrator, BDFIntegrator_Order3_CorrectCoefficients)
{
    const double dt = 0.1;
    SystemStateView state;
    state.dt = dt;

    std::vector<std::vector<Real>> hist_storage = {{Real(0.0)}, {Real(0.0)}, {Real(0.0)}};
    auto hist_spans = makeHistorySpans(hist_storage);
    state.u_history = hist_spans;

    BDFIntegrator bdf(3);
    const auto ctx = bdf.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());

    const Real inv_dt = 1.0 / static_cast<Real>(dt);
    const std::vector<Real> expected = {
        static_cast<Real>(11.0 / 6.0) * inv_dt,
        static_cast<Real>(-3.0) * inv_dt,
        static_cast<Real>(1.5) * inv_dt,
        static_cast<Real>(-1.0 / 3.0) * inv_dt,
    };
    expectStencilNear(ctx.dt1->a, expected, 1e-12);
}

TEST(TimeIntegrator, BDFIntegrator_Order4_CorrectCoefficients)
{
    const double dt = 0.2;
    SystemStateView state;
    state.dt = dt;

    std::vector<std::vector<Real>> hist_storage = {
        {Real(0.0)}, {Real(0.0)}, {Real(0.0)}, {Real(0.0)}};
    auto hist_spans = makeHistorySpans(hist_storage);
    state.u_history = hist_spans;

    BDFIntegrator bdf(4);
    const auto ctx = bdf.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());

    const Real inv_dt = 1.0 / static_cast<Real>(dt);
    const std::vector<Real> expected = {
        static_cast<Real>(25.0 / 12.0) * inv_dt,
        static_cast<Real>(-4.0) * inv_dt,
        static_cast<Real>(3.0) * inv_dt,
        static_cast<Real>(-4.0 / 3.0) * inv_dt,
        static_cast<Real>(0.25) * inv_dt,
    };
    expectStencilNear(ctx.dt1->a, expected, 1e-12);
}

TEST(TimeIntegrator, BDFIntegrator_Order5_CorrectCoefficients)
{
    const double dt = 0.3;
    SystemStateView state;
    state.dt = dt;

    std::vector<std::vector<Real>> hist_storage = {
        {Real(0.0)}, {Real(0.0)}, {Real(0.0)}, {Real(0.0)}, {Real(0.0)}};
    auto hist_spans = makeHistorySpans(hist_storage);
    state.u_history = hist_spans;

    BDFIntegrator bdf(5);
    const auto ctx = bdf.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());

    const Real inv_dt = 1.0 / static_cast<Real>(dt);
    const std::vector<Real> expected = {
        static_cast<Real>(137.0 / 60.0) * inv_dt,
        static_cast<Real>(-5.0) * inv_dt,
        static_cast<Real>(5.0) * inv_dt,
        static_cast<Real>(-10.0 / 3.0) * inv_dt,
        static_cast<Real>(1.25) * inv_dt,
        static_cast<Real>(-0.2) * inv_dt,
    };
    expectStencilNear(ctx.dt1->a, expected, 1e-11);
}

TEST(TimeIntegrator, BDFIntegrator_VariableStep_CorrectCoefficients)
{
    // Use moment conditions to validate variable-step weights for order 3.
    const double dt = 0.5;
    const double dt_prev = 0.25;
    const double dt_prev2 = 1.0;

    SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt_prev;
    std::vector<double> dt_hist = {dt_prev, dt_prev2};
    state.dt_history = dt_hist;

    std::vector<std::vector<Real>> hist_storage = {{Real(0.0)}, {Real(0.0)}, {Real(0.0)}};
    auto hist_spans = makeHistorySpans(hist_storage);
    state.u_history = hist_spans;

    BDFIntegrator bdf(3);
    const auto ctx = bdf.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    ASSERT_EQ(ctx.dt1->a.size(), 4u);

    const std::array<double, 4> nodes = {0.0, -dt, -(dt + dt_prev), -(dt + dt_prev + dt_prev2)};

    const auto& a = ctx.dt1->a;
    auto moment = [&](int p) -> double {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) {
            s += static_cast<double>(a[j]) * std::pow(nodes[j], p);
        }
        return s;
    };

    // Exactness for polynomials up to degree 3:
    // d/dt(1) = 0, d/dt(t) = 1, d/dt(t^2) = 0 at t=0, d/dt(t^3)=0 at t=0.
    EXPECT_NEAR(moment(0), 0.0, 1e-12);
    EXPECT_NEAR(moment(1), 1.0, 1e-12);
    EXPECT_NEAR(moment(2), 0.0, 1e-12);
    EXPECT_NEAR(moment(3), 0.0, 1e-12);
}

TEST(TimeIntegrator, BDFIntegrator_Order0_Throws)
{
    EXPECT_THROW((void)BDFIntegrator(0), svmp::FE::InvalidArgumentException);
}

TEST(TimeIntegrator, BDFIntegrator_Order6_Throws)
{
    EXPECT_THROW((void)BDFIntegrator(6), svmp::FE::InvalidArgumentException);
}

TEST(TimeIntegrator, BDFIntegrator_Name_ReturnsCorrectString)
{
    EXPECT_EQ(BDFIntegrator(1).name(), "BDF1");
    EXPECT_EQ(BDFIntegrator(2).name(), "BDF2");
    EXPECT_EQ(BDFIntegrator(5).name(), "BDF5");
}

TEST(TimeIntegrator, BackwardDifferenceIntegrator_BuildContext_Order1_Correct)
{
    SystemStateView state;
    state.dt = 0.5;

    BackwardDifferenceIntegrator integrator;
    const auto ctx = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    expectStencilNear(ctx.dt1->a, {2.0, -2.0}, 1e-12);
}

TEST(TimeIntegrator, BackwardDifferenceIntegrator_BuildContext_Order2_Correct)
{
    SystemStateView state;
    state.dt = 0.5;

    BackwardDifferenceIntegrator integrator;
    const auto ctx = integrator.buildContext(/*max_time_derivative_order=*/2, state);
    ASSERT_TRUE(ctx.dt2.has_value());
    expectStencilNear(ctx.dt2->a, {4.0, -8.0, 4.0}, 1e-12);
}

TEST(TimeIntegrator, BackwardDifferenceIntegrator_BuildContext_Order3_Correct)
{
    SystemStateView state;
    state.dt = 0.5;

    BackwardDifferenceIntegrator integrator;
    const auto ctx = integrator.buildContext(/*max_time_derivative_order=*/3, state);
    ASSERT_EQ(ctx.dt_extra.size(), 1u);
    ASSERT_TRUE(ctx.dt_extra[0].has_value());
    expectStencilNear(ctx.dt_extra[0]->a, {8.0, -24.0, 24.0, -8.0}, 1e-12);
}

TEST(TimeIntegrator, BDF2Integrator_BuildContext_VariableStep_CorrectCoefficients)
{
    SystemStateView state;
    state.dt = 0.5;
    state.dt_prev = 0.25;

    BDF2Integrator integrator;
    const auto ctx = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    ASSERT_EQ(ctx.dt1->a.size(), 3u);

    const Real dt = static_cast<Real>(state.dt);
    const Real r = static_cast<Real>(state.dt / state.dt_prev);
    const Real inv_dt = 1.0 / dt;
    const std::vector<Real> expected = {
        ((1.0 + 2.0 * r) / (1.0 + r)) * inv_dt,
        (-(1.0 + r)) * inv_dt,
        ((r * r) / (1.0 + r)) * inv_dt,
    };
    expectStencilNear(ctx.dt1->a, expected, 1e-12);
}

TEST(TimeIntegrator, BDF2Integrator_BuildContext_InsufficientHistory_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileBilinear((svmp::FE::forms::dt(u) * v).dx());
    sys.addCellKernel("op", u_field, u_field, std::make_shared<svmp::FE::forms::FormKernel>(std::move(ir)));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_n(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);

    SystemStateView state;
    state.dt = 0.5;
    state.u = u_n;
    state.u_prev = u_prev;
    // Intentionally omit u_prev2.

    TransientSystem transient(sys, std::make_shared<BDF2Integrator>());

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView A(static_cast<GlobalIndex>(n_dofs));
    A.zero();
    EXPECT_THROW((void)transient.assemble(req, state, &A, nullptr), svmp::FE::InvalidArgumentException);
}

TEST(TimeIntegrator, TimeIntegrator_ZeroTimeStep_Behavior)
{
    SystemStateView state;
    state.dt = 0.0;
    BackwardDifferenceIntegrator integrator;
    EXPECT_THROW((void)integrator.buildContext(/*max_time_derivative_order=*/1, state), svmp::FE::InvalidArgumentException);
}

TEST(TimeIntegrator, TimeIntegrator_NegativeTimeStep_Behavior)
{
    SystemStateView state;
    state.dt = -1.0;
    BackwardDifferenceIntegrator integrator;
    EXPECT_THROW((void)integrator.buildContext(/*max_time_derivative_order=*/1, state), svmp::FE::InvalidArgumentException);
}

TEST(TimeIntegrator, TimeIntegrator_MissingHistoryVectors_Throws)
{
    SystemStateView state;
    state.dt = 0.5;

    std::vector<std::vector<Real>> hist_storage = {{Real(0.0)}, {Real(0.0)}}; // only 2 history states
    auto hist_spans = makeHistorySpans(hist_storage);
    state.u_history = hist_spans;

    BDFIntegrator bdf3(3);
    EXPECT_THROW((void)bdf3.buildContext(/*max_time_derivative_order=*/1, state), svmp::FE::InvalidArgumentException);
}
