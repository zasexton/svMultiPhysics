/**
 * @file test_JacobianVerification.cpp
 * @brief Stronger Jacobian verification tests (Taylor remainder + central FD + Stokes coupling)
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/Vocabulary.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <cmath>
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

Real l2Norm(const std::vector<Real>& v)
{
    Real s = 0.0;
    for (const auto x : v) s += x * x;
    return std::sqrt(s);
}

std::vector<Real> matVec(const svmp::FE::assembly::DenseSystemView& A, const std::vector<Real>& x)
{
    std::vector<Real> y(static_cast<std::size_t>(A.numRows()), 0.0);
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        Real acc = 0.0;
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            acc += A.getMatrixEntry(i, j) * x[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] = acc;
    }
    return y;
}

} // namespace

TEST(StokesCouplingJacobianTest, CouplingBlocksMatchCentralDifferencesAndHaveCorrectTransposeSign)
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
    sys.addOperator("stokes_coupling");

    const auto u = svmp::FE::forms::TrialFunction(*u_space, "u");
    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");
    const auto p = svmp::FE::forms::TrialFunction(*p_space, "p");
    const auto q = svmp::FE::forms::TestFunction(*p_space, "q");

    const auto up_residual = (-p * svmp::FE::forms::div(v)).dx();
    const auto pu_residual = (q * svmp::FE::forms::div(u)).dx();

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 1, up_residual);
    blocks.setBlock(1, 0, pu_residual);

    (void)svmp::FE::systems::installResidualBlocks(
        sys, "stokes_coupling",
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
    req.op = "stokes_coupling";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    // Block structure sanity:
    //   [ 0   -B^T ]
    //   [ B     0  ]
    constexpr GlobalIndex n_u = 12;
    constexpr GlobalIndex n_p = 4;
    const Real zero_tol = 1e-14;
    const Real transpose_tol = 1e-12;

    for (GlobalIndex i = 0; i < n_u; ++i) {
        for (GlobalIndex j = 0; j < n_u; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), 0.0, zero_tol);
        }
    }
    for (GlobalIndex i = 0; i < n_p; ++i) {
        for (GlobalIndex j = 0; j < n_p; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(n_u + i, n_u + j), 0.0, zero_tol);
        }
    }
    for (GlobalIndex iu = 0; iu < n_u; ++iu) {
        for (GlobalIndex ip = 0; ip < n_p; ++ip) {
            const auto up = out.getMatrixEntry(iu, n_u + ip);
            const auto pu = out.getMatrixEntry(n_u + ip, iu);
            EXPECT_NEAR(up + pu, 0.0, transpose_tol);
        }
    }

    // Central-difference check: J(:,j) ~= (R(U+eps e_j) - R(U-eps e_j)) / (2 eps)
    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-6;
    const Real fd_tol = 1e-10;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "stokes_coupling";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;
        svmp::FE::systems::SystemStateView state_minus;
        state_minus.u = U_minus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        svmp::FE::assembly::DenseVectorView Rm(n_dofs);
        Rp.zero();
        Rm.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);
        (void)sys.assemble(req_vec, state_minus, nullptr, &Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, fd_tol);
        }
    }
}

TEST(NonlinearJacobianVerificationTest, ReverseADJacobianMatchesCentralDifferencesAndTaylorRemainderIsQuadratic)
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
    sys.addOperator("ns_reverse");

    const auto u = svmp::FE::forms::TrialFunction(*u_space, "u");
    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");
    const auto p = svmp::FE::forms::TrialFunction(*p_space, "p");
    const auto q = svmp::FE::forms::TestFunction(*p_space, "q");

    const Real nu = 0.01;
    const auto nu_c = svmp::FE::forms::FormExpr::constant(nu);

    const auto uu_residual =
        (svmp::FE::forms::inner(svmp::FE::forms::grad(u) * u, v) +
         nu_c * svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)))
            .dx();
    const auto up_residual = (-p * svmp::FE::forms::div(v)).dx();
    const auto pu_residual = (q * svmp::FE::forms::div(u)).dx();

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, uu_residual);
    blocks.setBlock(0, 1, up_residual);
    blocks.setBlock(1, 0, pu_residual);

    (void)svmp::FE::systems::installResidualBlocks(
        sys, "ns_reverse",
        {u_field, p_field},
        {u_field, p_field},
        blocks,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Reverse});

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
    req.op = "ns_reverse";
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

    // Central difference check.
    const Real eps_fd = 1e-6;
    const Real fd_tol = 2e-6;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "ns_reverse";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps_fd;
        U_minus[static_cast<std::size_t>(j)] -= eps_fd;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;
        svmp::FE::systems::SystemStateView state_minus;
        state_minus.u = U_minus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        svmp::FE::assembly::DenseVectorView Rm(n_dofs);
        Rp.zero();
        Rm.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);
        (void)sys.assemble(req_vec, state_minus, nullptr, &Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps_fd);
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, fd_tol);
        }
    }

    // Taylor remainder check:
    //   R(U + eps*dU) = R(U) + eps*J*dU + O(eps^2)
    std::vector<Real> dU(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        const Real x = static_cast<Real>(i + 1);
        dU[static_cast<std::size_t>(i)] = static_cast<Real>(0.2) * std::sin(static_cast<double>(0.37 * x));
    }

    const auto JdU = matVec(out, dU);

    const std::array<Real, 3> eps = {1e-2, 1e-3, 1e-4};
    std::array<Real, 3> remainder_norm{};
    for (std::size_t k = 0; k < eps.size(); ++k) {
        auto U_eps = U;
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            U_eps[static_cast<std::size_t>(i)] += eps[k] * dU[static_cast<std::size_t>(i)];
        }

        svmp::FE::systems::SystemStateView state_eps;
        state_eps.u = U_eps;

        svmp::FE::assembly::DenseVectorView R_eps(n_dofs);
        R_eps.zero();
        (void)sys.assemble(req_vec, state_eps, nullptr, &R_eps);

        std::vector<Real> remainder(static_cast<std::size_t>(n_dofs), 0.0);
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            remainder[idx] = (R_eps.getVectorEntry(i) - R0[idx]) - eps[k] * JdU[idx];
        }
        remainder_norm[k] = l2Norm(remainder);
    }

    ASSERT_GT(remainder_norm[0], 0.0);
    ASSERT_GT(remainder_norm[1], 0.0);
    ASSERT_GT(remainder_norm[2], 0.0);

    const auto slope01 = std::log(remainder_norm[1] / remainder_norm[0]) / std::log(eps[1] / eps[0]);
    const auto slope12 = std::log(remainder_norm[2] / remainder_norm[1]) / std::log(eps[2] / eps[1]);

    EXPECT_GT(slope01, 1.8);
    EXPECT_GT(slope12, 1.8);
}

TEST(StabilizedConvectionTest, SUPGJacobianMatchesCentralDifferencesAndTaylorRemainderIsQuadratic)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();

    auto u_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1,
                                           mesh,
                                           /*order=*/1,
                                           /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 1});
    sys.addOperator("supg");

    const auto u = svmp::FE::forms::TrialFunction(*u_space, "u");
    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");

    // Nonlinear, physics-agnostic convection + SUPG-style stabilization.
    // beta(u) = [u,u,u], tau(u) = 0.5*h / ||beta||, with safeNorm regularization.
    const auto beta = svmp::FE::forms::as_vector({u, u, u});
    const auto adv = svmp::FE::forms::inner(beta, svmp::FE::forms::grad(u));
    const auto tau = 0.5 * svmp::FE::forms::h() / svmp::FE::forms::safeNorm(beta);
    const auto residual = (adv * (v + tau * svmp::FE::forms::inner(beta, svmp::FE::forms::grad(v)))).dx();

    (void)svmp::FE::systems::installResidualForm(
        sys, "supg",
        u_field,
        u_field,
        residual,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Forward});

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    U = {0.2, 0.35, 0.15, 0.4};

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "supg";
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

    // Central difference check.
    const Real eps_fd = 1e-6;
    const Real fd_tol = 2e-8;

    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "supg";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps_fd;
        U_minus[static_cast<std::size_t>(j)] -= eps_fd;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;
        svmp::FE::systems::SystemStateView state_minus;
        state_minus.u = U_minus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        svmp::FE::assembly::DenseVectorView Rm(n_dofs);
        Rp.zero();
        Rm.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);
        (void)sys.assemble(req_vec, state_minus, nullptr, &Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps_fd);
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, fd_tol);
        }
    }

    // Taylor remainder check.
    std::vector<Real> dU(static_cast<std::size_t>(n_dofs), 0.0);
    dU = {0.25, -0.15, 0.05, -0.3};

    const auto JdU = matVec(out, dU);

    const std::array<Real, 3> eps = {1e-2, 1e-3, 1e-4};
    std::array<Real, 3> remainder_norm{};
    for (std::size_t k = 0; k < eps.size(); ++k) {
        auto U_eps = U;
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            U_eps[static_cast<std::size_t>(i)] += eps[k] * dU[static_cast<std::size_t>(i)];
        }

        svmp::FE::systems::SystemStateView state_eps;
        state_eps.u = U_eps;

        svmp::FE::assembly::DenseVectorView R_eps(n_dofs);
        R_eps.zero();
        (void)sys.assemble(req_vec, state_eps, nullptr, &R_eps);

        std::vector<Real> remainder(static_cast<std::size_t>(n_dofs), 0.0);
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            remainder[idx] = (R_eps.getVectorEntry(i) - R0[idx]) - eps[k] * JdU[idx];
        }
        remainder_norm[k] = l2Norm(remainder);
    }

    ASSERT_GT(remainder_norm[0], 0.0);
    ASSERT_GT(remainder_norm[1], 0.0);
    ASSERT_GT(remainder_norm[2], 0.0);

    const auto slope01 = std::log(remainder_norm[1] / remainder_norm[0]) / std::log(eps[1] / eps[0]);
    const auto slope12 = std::log(remainder_norm[2] / remainder_norm[1]) / std::log(eps[2] / eps[1]);

    EXPECT_GT(slope01, 1.8);
    EXPECT_GT(slope12, 1.8);
}
