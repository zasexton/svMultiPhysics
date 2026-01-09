/**
 * @file test_CoupledBoundaryConditionHelpers.cpp
 * @brief Unit tests for loop-free coupled boundary-condition helper APIs
 */

#include <gtest/gtest.h>

#include "Systems/CoupledBoundaryConditions.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include "Assembly/GlobalSystemView.h"

#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <span>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::Real;

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

} // namespace

TEST(CoupledBoundaryConditionHelpers, ApplyCoupledNeumann_UsesBoundaryFunctionalIntegral)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(/*boundary_marker=*/2);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("residual");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    svmp::FE::forms::BoundaryFunctional Q;
    Q.integrand = svmp::FE::forms::FormExpr::constant(1.0);
    Q.boundary_marker = 2;
    Q.name = "Q";
    Q.reduction = svmp::FE::forms::BoundaryFunctional::Reduction::Sum;

    const svmp::FE::constraints::CoupledNeumannBC bc(
        /*boundary_marker=*/2,
        std::vector<svmp::FE::forms::BoundaryFunctional>{Q},
        [](const svmp::FE::constraints::CoupledBCContext& ctx, Real, Real, Real) {
            return ctx.integrals.get("Q");
        });

    const std::vector<svmp::FE::constraints::CoupledNeumannBC> bcs = {bc};

    residual = svmp::FE::systems::bc::applyCoupledNeumann(
        sys,
        u_field,
        residual,
        v,
        std::span<const svmp::FE::constraints::CoupledNeumannBC>(bcs.data(), bcs.size()),
        /*aux=*/{},
        /*flux_name_prefix=*/"flux");

    svmp::FE::systems::installResidualForm(sys, "residual", u_field, u_field, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup(/*opts=*/{}, inputs);

    std::vector<Real> u_n(4, 0.0);
    svmp::FE::systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 1.0;
    state.u = std::span<const Real>(u_n.data(), u_n.size());

    svmp::FE::assembly::DenseVectorView rhs(4);
    rhs.zero();
    (void)sys.assembleResidual(state, rhs);

    const Real area = 0.5;
    const Real Qv = area;               // ∫ 1 ds over face {0,1,2}
    const Real expected = -Qv * area / 3.0;  // -∫ Q v ds, with Q constant == area

    EXPECT_NEAR(rhs.getVectorEntry(0), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(1), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(2), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(3), 0.0, 1e-12);
}

TEST(CoupledBoundaryConditionHelpers, ApplyCoupledNeumann_RegistersAuxStateAndResetsPerAssembly)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(/*boundary_marker=*/2);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("residual");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    svmp::FE::forms::BoundaryFunctional Q;
    Q.integrand = svmp::FE::forms::FormExpr::constant(1.0);
    Q.boundary_marker = 2;
    Q.name = "Q";
    Q.reduction = svmp::FE::forms::BoundaryFunctional::Reduction::Sum;

    // Auxiliary state X with evolution: X <- X + Q (per assembly). The manager resets
    // to the committed state each assembly call, so X does not drift across Newton iterations.
    svmp::FE::systems::AuxiliaryStateRegistration reg;
    reg.spec.size = 1;
    reg.spec.name = "X";
    reg.spec.associated_markers = {2};
    reg.initial_values = {0.0};
    reg.required_integrals = {Q};
    reg.rhs = [](const svmp::FE::systems::AuxiliaryState& /*state*/,
                 const svmp::FE::forms::BoundaryFunctionalResults& integrals,
                 Real /*t*/) {
        return integrals.get("Q");
    };

    const svmp::FE::constraints::CoupledNeumannBC bc(
        /*boundary_marker=*/2,
        std::vector<svmp::FE::forms::BoundaryFunctional>{Q},
        [](const svmp::FE::constraints::CoupledBCContext& ctx, Real, Real, Real) {
            return ctx.aux_state["X"];
        });

    const std::vector<svmp::FE::constraints::CoupledNeumannBC> bcs = {bc};
    const std::vector<svmp::FE::systems::AuxiliaryStateRegistration> aux = {reg};

    residual = svmp::FE::systems::bc::applyCoupledNeumann(
        sys,
        u_field,
        residual,
        v,
        std::span<const svmp::FE::constraints::CoupledNeumannBC>(bcs.data(), bcs.size()),
        std::span<const svmp::FE::systems::AuxiliaryStateRegistration>(aux.data(), aux.size()),
        /*flux_name_prefix=*/"flux");

    svmp::FE::systems::installResidualForm(sys, "residual", u_field, u_field, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup(/*opts=*/{}, inputs);

    std::vector<Real> u_n(4, 0.0);
    svmp::FE::systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 1.0;
    state.u = std::span<const Real>(u_n.data(), u_n.size());

    svmp::FE::assembly::DenseVectorView rhs0(4);
    rhs0.zero();
    (void)sys.assembleResidual(state, rhs0);

    svmp::FE::assembly::DenseVectorView rhs1(4);
    rhs1.zero();
    (void)sys.assembleResidual(state, rhs1);

    // Same committed state (no commit between calls) => identical result.
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(rhs1.getVectorEntry(i), rhs0.getVectorEntry(i), 1e-12);
    }

    // After commit, X is committed and the next assembly increments again.
    sys.commitTimeStep();

    svmp::FE::assembly::DenseVectorView rhs2(4);
    rhs2.zero();
    (void)sys.assembleResidual(state, rhs2);

    // The Neumann flux is X, and X doubles after commit.
    EXPECT_NEAR(rhs2.getVectorEntry(0), 2.0 * rhs0.getVectorEntry(0), 1e-12);
    EXPECT_NEAR(rhs2.getVectorEntry(1), 2.0 * rhs0.getVectorEntry(1), 1e-12);
    EXPECT_NEAR(rhs2.getVectorEntry(2), 2.0 * rhs0.getVectorEntry(2), 1e-12);
    EXPECT_NEAR(rhs2.getVectorEntry(3), 0.0, 1e-12);
}

TEST(CoupledBoundaryConditionHelpers, ApplyCoupledRobin_UsesBoundaryFunctionalIntegral)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(/*boundary_marker=*/2);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("residual");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    svmp::FE::forms::BoundaryFunctional Q;
    Q.integrand = svmp::FE::forms::FormExpr::constant(1.0);
    Q.boundary_marker = 2;
    Q.name = "Q";
    Q.reduction = svmp::FE::forms::BoundaryFunctional::Reduction::Sum;

    const svmp::FE::constraints::CoupledRobinBC bc(
        /*boundary_marker=*/2,
        std::vector<svmp::FE::forms::BoundaryFunctional>{Q},
        /*alpha=*/[](const svmp::FE::constraints::CoupledBCContext&, Real, Real, Real) { return 0.0; },
        /*beta=*/[](const svmp::FE::constraints::CoupledBCContext&, Real, Real, Real) { return 0.0; },
        /*g=*/[](const svmp::FE::constraints::CoupledBCContext& ctx, Real, Real, Real) { return ctx.integrals.get("Q"); });

    const std::vector<svmp::FE::constraints::CoupledRobinBC> bcs = {bc};

    residual = svmp::FE::systems::bc::applyCoupledRobin(
        sys,
        u_field,
        residual,
        u,
        v,
        std::span<const svmp::FE::constraints::CoupledRobinBC>(bcs.data(), bcs.size()),
        /*aux=*/{},
        /*alpha_name_prefix=*/"alpha",
        /*rhs_name_prefix=*/"rhs");

    svmp::FE::systems::installResidualForm(sys, "residual", u_field, u_field, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup(/*opts=*/{}, inputs);

    std::vector<Real> u_n(4, 0.0);
    svmp::FE::systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 1.0;
    state.u = std::span<const Real>(u_n.data(), u_n.size());

    svmp::FE::assembly::DenseVectorView rhs(4);
    rhs.zero();
    (void)sys.assembleResidual(state, rhs);

    const Real area = 0.5;
    const Real Qv = area;
    const Real expected = -Qv * area / 3.0;

    EXPECT_NEAR(rhs.getVectorEntry(0), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(1), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(2), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(3), 0.0, 1e-12);
}

TEST(CoupledBoundaryConditionHelpers, ApplyCoupledNeumann_ExpressionAware_BoundaryIntegralSymbol)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(/*boundary_marker=*/2);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("residual");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    const auto Q = svmp::FE::forms::FormExpr::boundaryIntegral(
        svmp::FE::forms::FormExpr::constant(1.0),
        /*marker=*/2,
        /*name=*/"Q");

    residual = svmp::FE::systems::bc::applyCoupledNeumann(
        sys,
        u_field,
        residual,
        v,
        /*boundary_marker=*/2,
        /*flux=*/Q);

    svmp::FE::systems::installResidualForm(sys, "residual", u_field, u_field, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup(/*opts=*/{}, inputs);

    std::vector<Real> u_n(4, 0.0);
    svmp::FE::systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 1.0;
    state.u = std::span<const Real>(u_n.data(), u_n.size());

    svmp::FE::assembly::DenseVectorView rhs(4);
    rhs.zero();
    (void)sys.assembleResidual(state, rhs);

    const Real area = 0.5;
    const Real expected = -(area * area) / 3.0;

    EXPECT_NEAR(rhs.getVectorEntry(0), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(1), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(2), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(3), 0.0, 1e-12);
}

TEST(CoupledBoundaryConditionHelpers, ApplyCoupledNeumann_ExpressionAware_AuxiliaryStateSymbol)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(/*boundary_marker=*/2);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("residual");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    svmp::FE::systems::AuxiliaryStateRegistration reg;
    reg.spec.size = 1;
    reg.spec.name = "X";
    reg.initial_values = {1.0};
    reg.rhs = [](const svmp::FE::systems::AuxiliaryState&,
                 const svmp::FE::forms::BoundaryFunctionalResults&,
                 Real) {
        return 0.0;
    };

    const auto flux = svmp::FE::forms::FormExpr::auxiliaryState("X");

    residual = svmp::FE::systems::bc::applyCoupledNeumann(
        sys,
        u_field,
        residual,
        v,
        /*boundary_marker=*/2,
        /*flux=*/flux,
        std::span<const svmp::FE::systems::AuxiliaryStateRegistration>(&reg, 1));

    svmp::FE::systems::installResidualForm(sys, "residual", u_field, u_field, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup(/*opts=*/{}, inputs);

    std::vector<Real> u_n(4, 0.0);
    svmp::FE::systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 1.0;
    state.u = std::span<const Real>(u_n.data(), u_n.size());

    svmp::FE::assembly::DenseVectorView rhs(4);
    rhs.zero();
    (void)sys.assembleResidual(state, rhs);

    const Real area = 0.5;
    const Real expected = -area / 3.0;

    EXPECT_NEAR(rhs.getVectorEntry(0), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(1), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(2), expected, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(3), 0.0, 1e-12);
}
