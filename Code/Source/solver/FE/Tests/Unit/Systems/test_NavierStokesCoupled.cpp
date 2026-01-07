/**
 * @file test_NavierStokesCoupled.cpp
 * @brief Coupled incompressible Navier–Stokes residual + Jacobian assembly tests
 *
 * This test validates that FE/Forms supports:
 * - vector-valued Trial/Test functions (via Spaces::ProductSpace)
 * - div/grad algebra needed by Navier–Stokes
 * - multi-field block assembly in Systems (u-u, u-p, p-u)
 * - AD-based Jacobians that match finite differences
 *
 * Stabilization terms (SUPG/PSPG/etc.) are intentionally excluded.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <memory>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
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

TEST(SpaceHelpersTest, MeshInferredSpacesRejectMixedCellTypes)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    EXPECT_THROW((void)svmp::FE::spaces::inferUniformElementType(*mesh),
                 svmp::FE::InvalidArgumentException);
    EXPECT_THROW((void)svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, /*order=*/1),
                 svmp::FE::InvalidArgumentException);

    EXPECT_EQ(svmp::FE::spaces::inferUniformElementType(*mesh, /*domain_id=*/1), ElementType::Tetra4);
    EXPECT_EQ(svmp::FE::spaces::inferUniformElementType(*mesh, /*domain_id=*/2), ElementType::Hex8);

    auto fluid = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1,
                                               mesh,
                                               /*order=*/1,
                                               /*components=*/3,
                                               /*domain_id=*/1);
    EXPECT_EQ(fluid->element_type(), ElementType::Tetra4);

    auto solid = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1,
                                         mesh,
                                         /*order=*/2,
                                         /*components=*/1,
                                         /*domain_id=*/2);
    EXPECT_EQ(solid->element_type(), ElementType::Hex8);

    EXPECT_THROW((void)svmp::FE::spaces::inferUniformElementType(*mesh, /*domain_id=*/42),
                 svmp::FE::InvalidArgumentException);
}

TEST(NavierStokesCoupledFormsTest, ResidualAndJacobianMatchFiniteDifferences)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();

    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1,
                                                 mesh,
                                                 /*order=*/1,
                                                 /*components=*/3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1,
                                           mesh,
                                           /*order=*/1,
                                           /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    // Block kernels (each residual has exactly one TrialFunction and one TestFunction).
    const auto u = svmp::FE::forms::TrialFunction(*u_space, "u");
    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");
    const auto p = svmp::FE::forms::TrialFunction(*p_space, "p");
    const auto q = svmp::FE::forms::TestFunction(*p_space, "q");

    const Real nu = 0.01;
    const auto nu_c = svmp::FE::forms::FormExpr::constant(nu);

    // Momentum: ( (u·∇)u, v ) + nu (∇u, ∇v)
    const auto uu_residual =
        (svmp::FE::forms::inner(svmp::FE::forms::grad(u) * u, v) +
         nu_c * svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)))
            .dx();

    // Pressure coupling: -(p, div(v))
    const auto up_residual = (-p * svmp::FE::forms::div(v)).dx();

    // Continuity: (q, div(u))
    const auto pu_residual = (q * svmp::FE::forms::div(u)).dx();

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, uu_residual);
    blocks.setBlock(0, 1, up_residual);
    blocks.setBlock(1, 0, pu_residual);

    (void)svmp::FE::systems::installResidualBlocks(
        sys, "ns",
        {u_field, p_field},
        {u_field, p_field},
        blocks,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Forward});

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 16);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.05) * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    // Finite-difference check: J(:,j) ~= (R(U+eps e_j) - R(U)) / eps
    const Real eps = 1e-7;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "ns";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(NavierStokesCoupledFormsTest, CoupledResidualInstallerMatchesFiniteDifferences)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();

    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1,
                                                 mesh,
                                                 /*order=*/1,
                                                 /*components=*/3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1,
                                           mesh,
                                           /*order=*/1,
                                           /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns_coupled");

    // Coupled residual blocks:
    // - momentum residual references both u and p state fields
    // - continuity residual references only u state field
    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_field, *p_space, "p");

    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");
    const auto q = svmp::FE::forms::TestFunction(*p_space, "q");

    const Real nu = 0.01;
    const auto nu_c = svmp::FE::forms::FormExpr::constant(nu);

    svmp::FE::forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(
        0,
        (svmp::FE::forms::inner(svmp::FE::forms::grad(u_state) * u_state, v) +
         nu_c * svmp::FE::forms::inner(svmp::FE::forms::grad(u_state), svmp::FE::forms::grad(v)) -
         p_state * svmp::FE::forms::div(v))
            .dx());
    residual.setBlock(1, (q * svmp::FE::forms::div(u_state)).dx());

    const std::array<svmp::FE::FieldId, 2> fields = {u_field, p_field};
    (void)svmp::FE::systems::installCoupledResidual(
        sys, "ns_coupled",
        fields,
        fields,
        residual,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Forward});

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 16);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.05) * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns_coupled";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "ns_coupled";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}
